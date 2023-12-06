import numpy as np
from collections import deque
import gtsam
from gtsam.symbol_shorthand import X,L,V,W
from mcf4pingpong.factors import *



class FactorGraph(gtsam.NonlinearFactorGraph):
    def __init__(self):
        super().__init__()
        self.X = set()
        self.L = set()
        self.V = set()
        self.W = set()
    def push_back(self,factor):
        super().push_back(factor)
        for k in factor.keys():
            if k < V(0):
                self.L.add(k - L(0))
            elif k < W(0):
                self.V.add(k - V(0))
            elif k < X(0):
                self.W.add(k - W(0))
            else:
                self.X.add(k - X(0))

initial_estimate = gtsam.Values()
graph = FactorGraph()
graph.push_back(TestFactor(gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-3),X(0), X(1), L(0), 0.0, 1.0))
graph.push_back(TestFactor(gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-3),X(0), X(2), L(2), 0.0, 1.0))
graph.push_back(TestFactor(gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-3),X(0), X(2), L(5), 0.0, 1.0))


print(graph.X)
print(graph.L)
print(graph.keys())

initial_estimate.insert(X(0), np.array([0,1,2]))
print(initial_estimate.exists(X(0)))