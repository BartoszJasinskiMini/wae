import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt
from EGO import EGO
from kriging import kriging
from true_function import true_function
from ga import ga


k = 2
n = 5*2

X = lhs(k, samples=n)
y = np.zeros((n, 1))

for i in range(k):
    y[i] = true_function(X[i], 1)

kr = kriging(k, X, y)

kr.train()

kr.plot_2d()

E = EGO(kr)
MinExpImp = 1e14
infill = 0

while abs(MinExpImp) > 1e-3 and infill < 3*n:
    Xnew, EI = E.next_infill()
    Ynew = true_function(Xnew, 1)
    kr.X = np.vstack((kr.X, Xnew))
    kr.y = np.vstack((kr.y, Ynew))
    infill = infill + 1

    kr.train()
    # kr.plot_2d()

