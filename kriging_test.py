import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt
from EGO import EGO
from kriging import kriging
from true_function import true_function
from ga import ga

#
# k = 2
# n = 5*2
#
# X = lhs(k, samples=n)
# y = np.zeros((n, 1))
#
# for i in range(k):
#     y[i] = true_function(X[i], 1)
#
# kr = kriging(k, X, y)
#
# kr.train()
#
# kr.plot_2d()
#
# E = EGO(kr)
# MinExpImp = 1e14
# infill = 0
#
# while abs(MinExpImp) > 1e-3 and infill < 3*n:
#     Xnew, EI = E.next_infill()
#     Ynew = true_function(Xnew, 1)
#     kr.X = np.vstack((kr.X, Xnew))
#     kr.y = np.vstack((kr.y, Ynew))
#     infill = infill + 1
#
#     kr.train()
#     # kr.plot_2d()
#
import numpy as np
from mipego import ParallelBO, ContinuousSpace, OrdinalSpace, NominalSpace, RandomForest

seed = 666
np.random.seed(seed)
dim_r = 2  # dimension of the real values

def obj_fun(x):
    x_r = np.array([x['continuous_%d'%i] for i in range(dim_r)])
    x_i = x['ordinal']
    x_d = x['nominal']
    _ = 0 if x_d == 'OK' else 1
    return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + _ * 2

# Continuous variables can be specified as follows:
# a 2-D variable in [-5, 5]^2
# for 2 variables, the naming scheme is continuous0, continuous1
C = ContinuousSpace([-5, 5], var_name='continuous') * dim_r

# Integer (ordinal) variables can be specified as follows:
# The domain of integer variables can be given as with continuous ones
# var_name is optional
I = OrdinalSpace([5, 15], var_name='ordinal')

# Discrete (nominal) variables can be specified as follows:
# No lb, ub... a list of categories instead
N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], var_name='nominal')

# The whole search space can be constructed:
search_space = C + I + N

# Bayesian optimization also uses a Surrogate model
# For mixed variable type, the random forest is typically used
model = RandomForest(levels=search_space.levels)

opt = ParallelBO(
    search_space=search_space,
    obj_fun=obj_fun,
    model=model,
    max_FEs=50,
    DoE_size=3,    # the initial DoE size
    eval_type='dict',
    acquisition_fun='MGFI',
    acquisition_par={'t' : 2},
    n_job=3,       # number of processes
    n_point=3,     # number of the candidate solution proposed in each iteration
    verbose=True   # turn this off, if you prefer no output
)
xopt, fopt, stop_dict = opt.run()

print('xopt: {}'.format(xopt))
print('fopt: {}'.format(fopt))
print('stop criteria: {}'.format(stop_dict))