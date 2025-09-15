import sys
import os
import time
from collections import defaultdict, deque
import math
import argparse
from readdata import ReadData
from EST import ESTHeuristic  # 用于产生 EST 上界
import numpy as np
import scipy.optimize as sco
from scipy.optimize import Bounds, LinearConstraint

class FJSP_CPLEX_Solver:
    def __init__(self):
        self.nop = 0  # 操作数量
        self.nmach = 0  # 机器数量
        self.arcs = 0  # 弧/约束数量
        self.prtime = []  # 处理时间矩阵
        self.dag = []  # 有向无环图（邻接表表示）
        self.Machs = []  # 每个机器能处理的操作集合
        
    def model_and_solve(self, input_file, timelimit=0, maxthreads=1, starts="", preset_L=None):
        """构建并求解MILP模型"""
        # 使用 readdata.ReadData 读取并准备数据
        rd = ReadData()
        rd.read_and_prepare_data(input_file)

        # 将 ReadData 中的数据转换为原有 solver 所需的结构
        self.nop = rd.n
        self.nmach = rd.numm
        self.arcs = rd.numA

        # 构建有向无环图（邻接表）dag表明了操作间后缀关系
        self.dag = [[] for _ in range(self.nop)]
        for v in range(self.nop):
            for w in range(self.nop):
                if rd.Adj[v][w] == 'A':
                    self.dag[v].append(w)

        # 构建处理时间矩阵和机器集合（仅把可行的机器加入 Machs）
        self.prtime = [[0] * self.nmach for _ in range(self.nop)] # 每个操作对应的机器开始时间初始化为0
        self.Machs = [set() for _ in range(self.nmach)] # 每台机器对应的可处理操作集合，初始化
        for v in range(self.nop):
            for M in range(self.nmach):
                if rd.Prt[v][M] < rd.infinite:
                    # 可行的处理时间
                    self.prtime[v][M] = int(rd.Prt[v][M])
                    self.Machs[M].add(v) # 记录机器 M 能处理的操作 v
                else:
                    # 不可行的机器设为 0（保持与原 read_input 行为一致）
                    self.prtime[v][M] = 0
        
        # 创建机器索引映射
        midx = [dict() for _ in range(self.nop)] # midx[v][M] = k 表示操作 v 在机器 M 上的索引为 j
        for v in range(self.nop):
            for M in range(self.nmach):
                if v in self.Machs[M]:
                    j = len(midx[v]) # 当前已有的机器数
                    midx[v][M] = j # 记录机器 M 在操作 v 上的索引
        
        # 创建软约束映射，指机器内顺序约束
        soft = {}
        for M in range(self.nmach):
            for v in range(self.nop - 1):
                if v not in self.Machs[M]:
                    continue
                for w in range(v + 1, self.nop):
                    if w not in self.Machs[M] or (v, w) in soft:
                        continue
                    # v != w, 都属于机器M，且尚未添加
                    k = len(soft)
                    soft[(v, w)] = k
                    soft[(w, v)] = k + 1
                    k += 2
        
        # 确定上界 L
        L = 0.0
        
        if starts:
            print("Warning: Scipy MILP does not support initial solutions. Ignoring starts parameter.")
            # 保留读取逻辑以计算 L，但不设置初始解
            with open(starts, 'r') as f:
                # 跳过前三行
                for _ in range(3):
                    f.readline()
                
                # 读取makespan
                line = f.readline().split()
                L = float(line[-1])
        else:
            # 若外部提供了 preset_L（例如来自 EST 启发式），优先使用之
            if preset_L is not None:
                L = float(preset_L)
            else:
                # 计算上界L（使用每个操作的最大处理时间之和作为上界）
                for v in range(self.nop):
                    x = 0.0
                    for M in range(self.nmach):
                        if v in self.Machs[M] and self.prtime[v][M] > x:
                            x = self.prtime[v][M]
                    L += x
        
        print(f"L = {L}")
        
        # 计算变量数量和索引
        num_s = self.nop # 开始时间变量数量
        num_c = self.nop # 完成时间变量数量
        num_x = len(soft) # 软约束变量数量
        num_f = sum(len(midx[v]) for v in range(self.nop)) # 机器可以分配的工序数量和
        total_vars = 1 + num_s + num_c + num_x + num_f  # z, s, c, x, f
        
        idx_z = 0 # z 变量索引
        idx_s = 1 # s 变量起始索引 对应文中s_{v,k}
        idx_c = idx_s + num_s # 对应t_{v,k}（完成时间，按机）
        idx_x = idx_c + num_c 
        idx_f = idx_x + num_x 
        
        # 对应文中：x_{v,k}（二进制，工序 v 分配到机器 k）
        f_indices = [] 
        f_start = idx_f
        for v in range(self.nop):
            f_indices.append([f_start + k for k in range(len(midx[v]))]) # 记录操作 v 的 f 变量索引
            f_start += len(midx[v])
        
        # 目标函数系数
        c_obj = np.zeros(total_vars)
        c_obj[idx_z] = 1  # minimize z
        
        # 变量界限 - 使用 Bounds 对象而不是元组列表
        lb = np.zeros(total_vars)  # 所有变量下界为0
        ub = np.full(total_vars, np.inf)  # 所有变量上界为无穷

        
        # 变量类型
        integrality = np.zeros(total_vars, dtype=int)
        # 设置 x 变量为二进制变量，对应文中y_{v,w,k}（在同一台机器 k 上 v 是否先于 w）初始化全为1
        for i in range(idx_x, idx_x + num_x):
            integrality[i] = 1  
        # 设置 f 变量为二进制变量 对应文中：x_{v,k}（二进制，工序 v 是否分配到机器 k）初始化全为1
        for v in range(self.nop):
            for idx in f_indices[v]:
                integrality[idx] = 1
        bounds = Bounds(lb, ub) #保证不来负数

        # 约束矩阵
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        
        # 约束(1): s[v] + c[v] <= z
        for v in range(self.nop):
            row = np.zeros(total_vars)
            row[idx_z] = -1
            row[idx_s + v] = 1
            row[idx_c + v] = 1
            A_ub.append(row)
            b_ub.append(0)
        
        # 约束(2): sum f_vars[v] == 1，确定一个工序只对应一个机器
        for v in range(self.nop):
            row = np.zeros(total_vars)
            for k in range(len(midx[v])):
                row[f_indices[v][k]] = 1
            A_eq.append(row)
            b_eq.append(1)
        
        # 约束(3): c[v] == sum prtime[v][M] * f_vars[v][k]
        for v in range(self.nop):
            row = np.zeros(total_vars)
            row[idx_c + v] = 1
            for M, k in midx[v].items():
                row[f_indices[v][k]] = -self.prtime[v][M]
            A_eq.append(row)
            b_eq.append(0)
        
        # 约束(4): 同一机器上的操作必须有顺序
        for M in range(self.nmach):
            for v in range(self.nop - 1):
                if v not in self.Machs[M]:
                    continue
                for w in range(v + 1, self.nop):
                    if w not in self.Machs[M]:
                        continue
                    i = midx[v][M]
                    j = midx[w][M]
                    k = soft[(v, w)]
                    kk = soft[(w, v)]
                    # 正确的 "至少一个顺序" 约束： x_vw + x_wv >= f_v + f_w - 1
                    # 转为 A_ub * x <= b_ub 的形式： -x_vw - x_wv + f_v + f_w <= 1
                    row = np.zeros(total_vars)
                    row[idx_x + k] = -1
                    row[idx_x + kk] = -1
                    row[f_indices[v][i]] = 1
                    row[f_indices[w][j]] = 1
                    A_ub.append(row)
                    b_ub.append(1)
        
        # 约束(5): 软约束的顺序关系
        for (v, w), idx in soft.items():
            row = np.zeros(total_vars)
            row[idx_s + v] = 1
            row[idx_c + v] = 1
            row[idx_x + idx] = L
            row[idx_s + w] = -1
            A_ub.append(row)
            b_ub.append(L)
        
        # 约束(6): 有向无环图的顺序约束
        for v in range(self.nop):
            for w in self.dag[v]:
                row = np.zeros(total_vars)
                row[idx_s + v] = 1
                row[idx_c + v] = 1
                row[idx_s + w] = -1
                A_ub.append(row)
                b_ub.append(0)
        
        # 转换为numpy数组
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # 构建约束对象
        constraints = []
        # 不等式约束：A_ub * x <= b_ub
        if A_ub.size > 0:
            constraints.append(LinearConstraint(A_ub, -np.inf * np.ones(len(b_ub)), b_ub))
        
        # 等式约束：A_eq * x == b_eq
        if A_eq.size > 0:
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
        
        # 求解参数
        start_time = time.time()
        options = {}
        if timelimit > 0:
            options['time_limit'] = timelimit

        # Debug checks before calling milp
        # 检查是否有操作没有可行机器分配
        empty_ops = [v for v in range(self.nop) if len(midx[v]) == 0]
        if empty_ops:
            print("不可行：以下操作没有任何可行机器分配（midx[v] 为空）：", empty_ops)
            return
        else:
            print("nice,牛啊！所有操作都有可行机器分配。")

        # 简单的 LP 松弛可行性检查（把 integrality 暂时设为 0）
        
        try:
            integrality_backup = integrality.copy()
            integrality_relax = np.zeros_like(integrality)
            print("正在检查 LP 松弛可行性（time_limit=10s）...")
            res_lp = sco.milp(c=c_obj, bounds=bounds, integrality=integrality_relax,
                              constraints=constraints, options={'time_limit': 10})
            if res_lp.success:
                print("LP 松弛有可行解：说明问题可能出在整数约束或建模逻辑（必须进一步检查 f/x 的互斥/等式）。")
            else:
                print("LP 松弛也不可行：说明线性约束或边界存在冲突，需要检查 A_eq/A_ub/b_eq/b_ub。")
        finally:
            integrality = integrality_backup
        # 调用milp求解
        res = sco.milp(c=c_obj, bounds=bounds, integrality=integrality, 
                      constraints=constraints, options=options)
        solve_time = time.time() - start_time
        
        # 输出结果
        if res.success:
            print(f"Solution found in {solve_time:.2f} seconds")
            print(f"Makespan (z): {res.x[idx_z]:.2f}")
            
            # 输出开始时间
            for v in range(self.nop):
                print(f"s_{{{v}}} = {res.x[idx_s + v]:.2f}")
            
            # 输出完成时间
            for v in range(self.nop):
                print(f"c_{{{v}}} = {res.x[idx_c + v]:.2f}")
            
            # 输出机器分配
            for v in range(self.nop):
                for M in range(self.nmach):
                    if v in self.Machs[M]:
                        k = midx[v][M]
                        if res.x[f_indices[v][k]] >= 0.99:  # 考虑浮点误差
                            print(f"f_{{{v}, {M}}} = 1")
            
            # 输出软约束变量
            for (v, w), idx in soft.items():
                if res.x[idx_x + idx] >= 0.99:
                    print(f"x_{{{v}, {w}}} = 1")
        else:
            print("No feasible solution found")
            print(f"Status: {res.status}")
            print(f"Message: {res.message}")

def main():
    """主函数：使用 argparse 解析参数"""
    parser = argparse.ArgumentParser(description="FJSP Scipy MILP solver")
    # 将位置参数改为可选（nargs='?'），并使用原始字符串避免转义警告
    parser.add_argument("input_file", nargs='?', default=r'E:\调度问题\巴西人\MFJS08.txt', help="输入文件路径")
    parser.add_argument("-t", "--timelimit", type=int, default=3600, help="求解时间上限（秒），默认 0 表示不限制")
    parser.add_argument("-p", "--maxthreads", type=int, default=1, help="最大线程数（scipy不支持，直接忽略），默认 1")
    parser.add_argument("-s", "--startsol", default="", help="始解文件路径（scipy不支持初始解，直接忽略）")
    args = parser.parse_args()

    input_file = args.input_file
    timelimit = args.timelimit
    maxthreads = args.maxthreads
    startsol = args.startsol

    if timelimit > 0:
        print(f"Time limit set to {timelimit}")
    if maxthreads > 0:
        print(f"Maximum threads set to {maxthreads} (ignored in scipy)")
    if startsol:
        print(f"Start solution file: {startsol} (ignored in scipy)")

    # 尝试使用 EST 启发式计算上界作为 preset_L（仅在没有外部 startsol 时使用）
    preset_L = None
    try:
        est = ESTHeuristic()
        est.text = input_file
        est.run()
        preset_L = est.mks
        print(f"Using EST makespan as preset_L = {preset_L}")
    except SystemExit:
        # EST 内部可能因错误退出，若发生则不阻止后续 MILP 求解，继续使用默认上界估计
        print("EST failed or exited; falling back to default L estimation")
    except Exception as e:
        print(f"EST heuristic failed: {e}; falling back to default L estimation")

    solver = FJSP_CPLEX_Solver()
    solver.model_and_solve(input_file, timelimit, maxthreads, startsol, preset_L=preset_L)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())