import sys
import os
import time
from collections import defaultdict, deque
import math
import argparse
from readdata import ReadData
from EST import ESTHeuristic  # 用于产生 EST 上界
import numpy as np
# 移除 scipy.optimize 相关导入，改用 PySCIPOpt
# import scipy.optimize as sco
# from scipy.optimize import Bounds, LinearConstraint
from pyscipopt import Model, quicksum

class FJSP_CPLEX_Solver:
    def __init__(self):
        self.nop = 0  # 操作数量
        self.nmach = 0  # 机器数量
        self.arcs = 0  # 弧/约束数量
        self.prtime = []  # 处理时间矩阵
        self.dag = []  # 有向无环图（邻接表表示）
        self.Machs = []  # 每个机器能处理的操作集合
        
    def model_and_solve(self, input_file, timelimit=0, maxthreads=1, starts="", preset_L=None):
        """构建并求解MILP模型（使用 PySCIPOpt）"""
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
        
        # 创建软约束映射，指机器内顺序约束。现在没有考虑先后关系，只是在记录每个机器上操作对
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
            print("Warning: PySCIPOpt 初始解逻辑未实现，忽略 starts 参数。")
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
        
        #这是在检查数据的可行性，即原始数据是否有错误
        # 检查是否有操作没有可行机器分配
        empty_ops = [v for v in range(self.nop) if len(midx[v]) == 0]
        if empty_ops:
            print("不可行：以下操作没有任何可行机器分配（midx[v] 为空）：", empty_ops)
            return
        else:
            print("nice,牛啊！所有操作都有可行机器分配。")

        # 使用 PySCIPOpt 构建模型
        model = Model("FJSP_PySCIPOpt")
        # 可选设置
        if timelimit and timelimit > 0:
            model.setParam('limits/time', float(timelimit))
            print(f"Time limit set to {timelimit}")
        if maxthreads and maxthreads > 0:
            try:
                model.setParam('threads', int(maxthreads))
                print(f"Maximum threads set to {maxthreads}")
            except Exception:
                print("设置线程数失败或未生效（PySCIPOpt 参数可能不同），已忽略。")

        # 变量
        z = model.addVar("z", vtype="C", lb=0.0) # 目标 makespan 最小总时间
        s_vars = [model.addVar(f"s_{v}", vtype="C", lb=0.0) for v in range(self.nop)] # s 开始时间变量 已经命令下限为0，不会出现负数。15，16约束实现。
        c_vars = [model.addVar(f"c_{v}", vtype="C", lb=0.0) for v in range(self.nop)] # c 操作时间变量
        # x 二进制变量，按 soft 索引创建,对应文中y_{v,w,k}（在同一台机器 k 上 v 是否先于 w）
        num_x = len(soft)
        x_vars = [None] * num_x
        for (v, w), idx in soft.items():
            # 可能被重复赋值两次，但指向相同 idx 的赋值结果相同
            if x_vars[idx] is None:
                x_vars[idx] = model.addVar(f"x_{v}_{w}", vtype="B") #vtype="B"表示以二进制储存，没有设置初始变量；在调用 optimize() 之前变量没有数值意义。
        # f 分配二进制变量,对应文中：x_{v,k}（二进制，工序 v 是否分配到机器 k）
        f_vars = []
        for v in range(self.nop):
            fv = []
            for k in range(len(midx[v])):
                fv.append(model.addVar(f"f_{v}_{k}", vtype="B"))
            f_vars.append(fv)

        # 目标：最小化 z
        model.setObjective(z, "minimize")

        # 约束(1): s[v] + c[v] <= z 此约束也完成了 s[v] <= z 的隐含约束,即英文中的10约束
        for v in range(self.nop):
            model.addCons(s_vars[v] + c_vars[v] <= z)

        # 约束(2): 每个工序分配到且仅到一台机器 约束9
        for v in range(self.nop):
            model.addCons(quicksum(f_vars[v][k] for k in range(len(midx[v]))) == 1)

        # 约束(3): c[v] == sum prtime[v][M] * f_vars[v][k] 通过约束操作时间，间接实现了约束12
        for v in range(self.nop):
            expr = quicksum(self.prtime[v][M] * f_vars[v][k] for M, k in midx[v].items())
            model.addCons(c_vars[v] == expr)

        # 约束(4): 同一机器上的操作必须有顺序： -x_vw - x_wv + f_v + f_w <= 1 实现约束11
        for M in range(self.nmach):
            for v in range(self.nop - 1):
                if v not in self.Machs[M]:
                    continue
                for w in range(v + 1, self.nop):#f_v 和 f_w 都为1，表示在同一个机器上。
                    if w not in self.Machs[M]:
                        continue
                    i = midx[v][M]
                    j = midx[w][M]
                    k = soft[(v, w)]
                    kk = soft[(w, v)]
                    model.addCons(- x_vars[k] - x_vars[kk] + f_vars[v][i] + f_vars[w][j] <= 1)

        # 约束(5): 软约束的顺序关系: s_v + c_v + L*x_idx - s_w <= L 约束13
        for (v, w), idx in soft.items():
            model.addCons(s_vars[v] + c_vars[v] + L * x_vars[idx] - s_vars[w] <= L)

        # 约束(6): DAG 的顺序约束: s_v + c_v - s_w <= 0，满足前序工序的完成时间不超过后续工序的开始时间，约束14
        for v in range(self.nop):
            for w in self.dag[v]:
                model.addCons(s_vars[v] + c_vars[v] - s_vars[w] <= 0)

        # 求解
        model.optimize()

        status = model.getStatus()
        # getStatus() 在 PySCIPOpt 中返回字符串，使用字符串判断更稳健
        status_str = str(status).lower()
        if ("optimal" in status_str) or ("feasible" in status_str):
            solve_time = model.getSolvingTime()
            print(f"Solution found in {solve_time:.2f} seconds")
            z_val = model.getVal(z)
            print(f"Makespan (z): {z_val:.2f}")

            # 输出开始时间
            for v in range(self.nop):
                print(f"s_{{{v}}} = {model.getVal(s_vars[v]):.2f}")
            
            # 输出每个操作的完成时间
            for v in range(self.nop):
                print(f"c_{{{v}}} = {model.getVal(c_vars[v]):.2f}")
            
            # 输出机器分配
            for v in range(self.nop):
                for M in range(self.nmach):
                    if v in self.Machs[M]:
                        k = midx[v][M]
                        if model.getVal(f_vars[v][k]) >= 0.5:
                            print(f"f_{{{v}, {M}}} = 1")

            # 输出软约束变量
            for (v, w), idx in soft.items():
                if model.getVal(x_vars[idx]) >= 0.5:
                    print(f"x_{{{v}, {w}}} = 1")
                    
            # 按机器汇总操作信息
            machine_operations = defaultdict(list)
            for v in range(self.nop):
                for M in range(self.nmach):
                    if v in self.Machs[M]:
                        k = midx[v][M]
                        if model.getVal(f_vars[v][k]) >= 0.5:
                            start_time = model.getVal(s_vars[v])
                            proc_time = model.getVal(c_vars[v])
                            machine_operations[M].append({
                                'operation': v,
                                'start_time': start_time,
                                'proc_time': proc_time,
                                'end_time': start_time + proc_time
                            })

            # 按机器编号输出排序后的操作安排
            print("\n=== 机器工序安排 ===")
            for M in sorted(machine_operations.keys()):
                # 按开始时间排序
                sorted_ops = sorted(machine_operations[M], key=lambda x: x['start_time'])
                
                print(f"\n机器 {M} 的工序安排:")
                print("工序\t开始时间\t处理时间\t结束时间")
                for op in sorted_ops:
                    print(f"{op['operation']}\t{op['start_time']:.2f}\t{op['proc_time']:.2f}\t{op['end_time']:.2f}")
                    
            #输出总时间
            print(f"Makespan (z): {z_val:.2f}")
        else:
            print("No feasible solution found")
            print(f"Status: {status}")

def main():
    """主函数：使用 argparse 解析参数"""
    parser = argparse.ArgumentParser(description="FJSP PySCIPOpt MILP solver")
    parser.add_argument("input_file", nargs='?', default=r'E:\调度问题\巴西人\brazil-man\YFJS01.txt', help="输入文件路径")
    parser.add_argument("-t", "--timelimit", type=int, default=3600, help="求解时间上限（秒）")
    parser.add_argument("-p", "--maxthreads", type=int, default=1, help="最大线程数")
    parser.add_argument("-s", "--startsol", default="", help="始解文件路径（忽略）")
    args = parser.parse_args()

    input_file = args.input_file
    timelimit = args.timelimit
    maxthreads = args.maxthreads
    startsol = args.startsol

    if timelimit > 0:
        print(f"Time limit set to {timelimit}")
    if maxthreads > 0:
        print(f"Maximum threads set to {maxthreads} (may be ignored)")

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