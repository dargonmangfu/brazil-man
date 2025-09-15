import sys
import os
import numpy as np
from collections import deque
from readdata import ReadData
# 定义常量
INPUTBUFFERSIZE = 128

class ESTHeuristic:
    def __init__(self):
        self.n = 0  # 顶点/操作数量
        self.numA = 0  # 弧/约束数量
        self.numm = 0  # 机器数量
        self.Adj = None  # 邻接矩阵
        self.Prt = None  # 处理时间矩阵
        self.infinite = 0  # 表示无穷大的值
        self.uu = None  # 拓扑排序
        self.ff = None  # 机器分配
        self.st = None  # 开始时间
        self.ct = None  # 完成时间
        self.mks = 0  # 最大完成时间（makespan）
        self.text = "sfjs01.txt"  # 默认输入文件名
    
    def est_heuristic(self):
        # 计算平均处理时间（与 C 版本对应）
        meanprt = np.zeros(self.n, dtype=float)
        for v in range(self.n):
            howmany = 0
            total = 0.0
            for k in range(self.numm):
                if self.Prt[v][k] < self.infinite:
                    total += self.Prt[v][k]
                    howmany += 1
            meanprt[v] = total / howmany if howmany > 0 else 0.0

        # 计算从每个顶点出发的最长剩余路径
        lpfrom = self.longest_paths_from(meanprt)
        for w in range(self.n):
            lpfrom[w] += meanprt[w]

        # 初始化数组（与 C 代码语义对应）
        U = np.zeros(self.n, dtype=int)  # 已调度的顶点标记
        self.uu = np.full(self.n, -99999, dtype=int)  # 调度顺序占位（便于调试）
        self.ff = np.zeros(self.n, dtype=int)  # 机器分配
        self.st = np.zeros(self.n, dtype=int)  # 开始时间
        self.ct = np.zeros(self.n, dtype=int)  # 完成时间
        avl = np.zeros(self.numm, dtype=int)  # 机器可用时间
        rdy = np.zeros(self.n, dtype=int)  # 操作就绪时间
        self.mks = 0

        # 主循环：选择 n 次
        for q in range(self.n):
            # 计算每个未调度操作的就绪时间
            for w in range(self.n):
                if U[w] == 0:
                    rdy[w] = 0
                    # 检查已调度前驱的完成时间
                    for i in range(q):
                        if self.Adj[self.uu[i]][w] == 'A':
                            rdy[w] = max(rdy[w], self.ct[i])
                    # 若存在未调度的前驱，则该操作尚未就绪
                    for v in range(self.n):
                        if U[v] == 0 and self.Adj[v][w] == 'A':
                            rdy[w] = self.infinite
                    # 若 rdy[w] == infinite 则表示 w 未就绪

            # 寻找最佳候选操作和机器（与 C 逻辑一致）
            earlst = self.infinite
            maxlpfrom = -1.0
            xx = -1
            ll = -1

            for w in range(self.n):
                if U[w] == 1 or rdy[w] == self.infinite:
                    continue
                for k in range(self.numm):
                    if self.Prt[w][k] >= self.infinite:
                        continue
                    t = max(rdy[w], avl[k])
                    if earlst > t or (earlst == t and maxlpfrom < lpfrom[w]):
                        earlst = t
                        xx = w
                        ll = k
                        maxlpfrom = lpfrom[w]

            # 若未找到可调度顶点，说明输入有问题（如环）
            if xx == -1:
                print("*** No schedulable vertex found (graph may have cycles or bad data)")
                sys.exit(1)

            # 调度选中的操作和机器
            U[xx] = 1
            self.uu[q] = xx
            self.ff[q] = ll
            self.st[q] = earlst
            self.ct[q] = earlst + self.Prt[xx][ll]
            avl[ll] = self.ct[q]
            self.mks = max(self.mks, self.ct[q])
    
    #巴西人用了反向拓扑排序
    def longest_paths_from(self, meanprt):
        """
        按照 C 代码 LongestPathsFrom 的语义实现。
        输入 meanprt: 长度为 n 的平均处理时间数组（numpy）
        返回 lpfrom: 长度为 n 的 numpy 浮点数组，表示从每个顶点出发的最长剩余路径（不包含自身 meanprt）
        若图包含环则打印错误并退出。
        """
        # 计算出度并初始化队列（所有出度为 0 的顶点）
        outdeg = np.zeros(self.n, dtype=int)
        queue = deque()
        for v in range(self.n):
            cnt = 0
            for w in range(self.n):
                if self.Adj[v][w] == 'A':
                    cnt += 1
            outdeg[v] = cnt
            if cnt == 0:
                queue.append(v)

        # 初始化 lpfrom
        lpfrom = np.zeros(self.n, dtype=float)

        # 处理队列：从终点回溯更新前驱的 lpfrom
        processed = 0
        while queue:
            w = queue.popleft()
            processed += 1
            # 对所有可能的前驱 v，若存在边 v->w 则尝试更新 lpfrom[v]
            for v in range(self.n):
                if self.Adj[v][w] != 'A':
                    continue
                candidate = lpfrom[w] + meanprt[w]
                if lpfrom[v] < candidate:
                    lpfrom[v] = candidate
                outdeg[v] -= 1
                if outdeg[v] == 0:
                    queue.append(v)

        # 若未处理所有顶点，说明存在环
        if processed < self.n:
            print("*** Graph contains a cycle or bad data (cannot compute longest paths from sinks)")
            sys.exit(1)

        return lpfrom
    
    def print_adj(self):
        """
        每行以 "v:" 开头，后跟所有与 v 相连的 w。
        """
        for v in range(self.n):
            line = f"{v:2d}:"
            for w in range(self.n):
                if self.Adj[v][w] != ' ':
                    line += f" {w:2d}"
            print(line)
        sys.stdout.flush()
    
    def run(self):
        # 使用 ReadData 正确读取并把字段复制到当前 ESTHeuristic 实例
        rd = ReadData()
        rd.read_and_prepare_data(self.text)
        self.n = rd.n
        self.numA = rd.numA
        self.numm = rd.numm
        self.Adj = rd.Adj
        self.Prt = rd.Prt
        self.infinite = rd.infinite
         # 调用启发式和输出结果
        self.est_heuristic()
        #self.print_adj()
        '''
        print(f"mks={self.mks}")
        print("\n i uu[i] ff[i] st[i] ct[i]")
        for i in range(self.n):
            print(f"{i:2d} {self.uu[i]:5d} {self.ff[i]:5d} {self.st[i]:5d} {self.ct[i]:5d}")
        '''

# 主程序
if __name__ == "__main__":
    est = ESTHeuristic()
    est.run()