import sys
import os
import numpy as np

class ReadData:
    def __init__(self):
        self.n = 0  # 顶点/操作数量
        self.numA = 0  # 弧/约束数量
        self.numm = 0  # 机器数量
        self.Adj = None  # 邻接矩阵 n*n
        self.Prt = None  # 处理时间矩阵
        self.infinite = 0  
    
    def read_and_prepare_data(self, filename="MK01.txt"):
        # 打开 MK01.txt 并按 C 实现的行为读取数据（跳过注释、逐工序读取 M + M*(machine,time)）
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, filename)
        try:
            f = open(path, "r", encoding="utf-8")
        except Exception:
            print(f"*** {path} not found or cannot be opened")
            sys.exit(1)

        def next_data_line():
            while True:
                line = f.readline()
                if not line:
                    return None
                # 去掉行尾/行内注释（'#' 及之后部分），再 strip
                if '#' in line:
                    line = line.split('#', 1)[0]
                s = line.strip()
                if not s:
                    continue
                return s

        # 读取头部 N A K
        line = next_data_line()
        if line is None:
            print("*** Unexpected end of file when reading header")
            sys.exit(1)
        parts = line.split()
        if len(parts) < 3:
            print("*** Bad header line")
            sys.exit(1)
        self.n = int(parts[0])
        self.numA = int(parts[1])
        self.numm = int(parts[2])

        # 初始化邻接矩阵
        self.Adj = np.full((self.n, self.n), ' ', dtype=str)
        # 同时准备邻接表、入度、以及机器->操作集合（与 C 版对应）
        self.dag = [[] for _ in range(self.n)] # self.dag[0]存储与节点0相连的所有节点编号。
        indegree = [0] * self.n
        self.Machs = [set() for _ in range(self.numm)]

        # 读取弧/约束
        count = 0
        while count < self.numA:
            line = next_data_line()
            if line is None:
                print("*** Unexpected end of file when reading arcs")
                sys.exit(1)
            parts = line.split()
            if len(parts) < 2:
                continue
            v = int(parts[0]) # v是起点
            w = int(parts[1]) # w是终点
            if v == w:
                print("*** No loops, please!")
                sys.exit(1)
            if self.Adj[v][w] == 'A':
                print(f"*** Repeated arc {v}-{w}")
                sys.exit(1)
            self.Adj[v][w] = 'A'
            self.dag[v].append(w)
            indegree[w] += 1
            count += 1

        # 不可行的机器操作时间设为 0（保持与原 read_input 行为一致）
        self.Prt = np.full((self.n, self.numm), 0, dtype=int)

        # 逐工序读取处理时间：每行以 hm 开头，随后 hm 对 (machine,time)
        for v in range(self.n):
            line = next_data_line()
            if line is None:
                print("*** Unexpected end of file when reading processing times")
                sys.exit(1)
            parts = line.split()
            if len(parts) < 1:
                print("*** Bad processing time line")
                sys.exit(1)
            hm = int(parts[0]) # hm是工序v可处理的机器数
            if hm <= 0:
                print(f"*** No machine specified for {v}")
                sys.exit(1)
            # required = 1 + 2 * hm
            # # 需要时继续读取后续非注释行直到收集够 token
            # while len(parts) < required:
            #     more = f.readline()
            #     if not more:
            #         print("*** Unexpected end of file when reading processing times")
            #         sys.exit(1)
            #     more = more.strip()
            #     if not more or more.startswith('#'):
            #         continue
            #     parts.extend(more.split())

            for j in range(hm):
                mchnv = int(parts[1 + 2 * j])
                prtv = int(parts[1 + 2 * j + 1])
                if mchnv < 0 or mchnv >= self.numm:
                    print(f"*** Machine {mchnv} out of range")
                    sys.exit(1)
                if prtv <= 0:
                    print("*** Processing times must be positive")
                    sys.exit(1)
                self.Prt[v][mchnv] = prtv
                # 与 C 版 readInput 中相似：记录机器可执行的操作
                self.Machs[mchnv].add(v)

        # 计算 infinite = 0 + sum(每个顶点的最大处理时间)
        self.infinite = 0
        for v in range(self.n): #取每个工序中需要的最多的加工时间
            largest = 0
            for k in range(self.numm):
                if self.Prt[v][k] > largest:
                    largest = self.Prt[v][k]
            self.infinite += largest

        # 简单的上界检查
        if self.infinite > (sys.maxsize // 100):
            print("*** Sum of processing times too large")
            sys.exit(1)

        # 将未指定的处理时间设置为 infinite 处理Prt矩阵中的 -1
        # for v in range(self.n):
        #     for k in range(self.numm):
        #         if self.Prt[v][k] == -1:
        #             self.Prt[v][k] = self.infinite
        # 计算入度为0的 heads，类似 C 中 readInput2 的 heads 输出
        self.heads = [v for v in range(self.n) if indegree[v] == 0]

        f.close()