#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A short and simple example experiment with restarts.

The script is fully functional but also emphasises on readability. It
features restarts, timings and recording termination conditions.

To benchmark a different solver, `fmin` must be re-assigned and another
`elif` block added around line 119 to account for the solver-specific
call.

When calling the script, previously assigned variables can be re-assigned
via a ``name=value`` argument without white spaces, where ``value`` is
interpreted as a single python literal. Additionally, ``batch`` is recognized
as argument defining the `current_batch` number and the number of `batches`,
like ``batch=2/8`` runs batch 2 of 8.

Examples, preceeded by "python" in an OS shell and by "run" in an IPython
shell::

    example_experiment2.py budget_multiplier=3  # times dimension

    example_experiment2.py budget_multiplier=1e4 cocopp=None  # omit post-processing
    
    example_experiment2.py budget_multiplier=1e4 suite_name=bbob-biobj

    example_experiment2.py budget_multiplier=1000 batch=1/16

Post-processing with `cocopp` is only invoked in the single-batch case.

Details: ``batch=9/8`` is equivalent to ``batch=1/8``. The first number
is taken modulo to the second.

See the code: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment2.py>`_

See a beginners example experiment: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment_for_beginners.py>`_

"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and ..."
import sys
import time  # output some timings per evaluation
from collections import defaultdict
import os, webbrowser  # to show post-processed results in the browser
import numpy as np  # for median, zeros, random, asarray
import cocoex  # experimentation module
import mocmaes
from pyDOE import lhs
# from smt.applications import EGO
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
from EGO import EGO
from mipego import ParallelBO, ContinuousSpace, OrdinalSpace, NominalSpace, RandomForest, BO
from mipego import OptimizerPipeline, BO, ContinuousSpace, Solution, GaussianProcess
from mipego.optimizer import OnePlusOne_Cholesky_CMA
from kriging import kriging
from true_function import true_function
from mipego.GaussianProcess.trend import constant_trend

try: import cocopp  # post-processing module
except: pass

### MKL bug fix
def set_num_threads(nt=1, disp=1):
    """see https://github.com/numbbo/coco/issues/1919
    and https://twitter.com/jeremyphoward/status/1185044752753815552
    """
    try: import mkl
    except ImportError: disp and print("mkl is not installed")
    else:
        mkl.set_num_threads(nt)
    nt = str(nt)
    for name in ['OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS',
                 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS']:
        os.environ[name] = nt
    disp and print("setting mkl threads num to", nt)

if sys.platform.lower() not in ('darwin', 'windows'):
    set_num_threads(1)

### solver imports (add other imports if necessary)
import scipy.optimize  # to define the solver to be benchmarked
try: import cma
except: pass  # may not be installed

def random_search(f, lbounds, ubounds, evals):
    """Won't work (well or at all) for `evals` much larger than 1e5"""
    [f(x) for x in np.asarray(lbounds) + (np.asarray(ubounds) - lbounds)
                               * np.random.rand(int(evals), len(ubounds))]

### input (to be modified if necessary/desired)
# fmin = scipy.optimize.fmin
#fmin = scipy.optimize.fmin_slsqp
# fmin = scipy.optimize.fmin_cobyla
#fmin = cocoex.solvers.random_search
# fmin = mocmaes.MOCMAES
fmin = EGO
# fmin = cma.fmin2

suite_name = "bbob-biobj"  # see cocoex.known_suite_names
budget_multiplier = 10  # times dimension, increase to 10, 100, ...
suite_filter_options = (""  # without filtering, a suite has instance_indices 1-15
                        # "dimensions: 2,3,5,10,20 "  # skip dimension 40
                        # "instance_indices: 1-5 "  # relative to suite instances
                       )
# for more suite filter options see http://numbbo.github.io/coco-doc/C/#suite-parameters
suite_year_option = ""  # "year: 2022"  # determine instances by year, not all years work for all suites :-(

batches = 1  # number of batches, batch=3/32 works to set both, current_batch and batches
current_batch = 1  # only current_batch modulo batches is relevant
output_folder = ''

### possibly modify/overwrite above input parameters from input args
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', 'help', '-help', '--help'):
        print(__doc__)
        raise ValueError("printed help and aborted")
    input_params = cocoex.utilities.args_to_dict(
        sys.argv[1:], globals(), {'batch': 'current_batch/batches'}, print=print)
    globals().update(input_params)  # (re-)assign variables

# extend output folder input parameter, comment out if desired otherwise
output_folder += '%s_of_%s_%dD_on_%s' % (
        fmin.__name__, fmin.__module__, int(budget_multiplier), suite_name)

if batches > 1:
    output_folder += "_batch%03dof%d" % (current_batch, batches)

### prepare
suite = cocoex.Suite(suite_name, suite_year_option, suite_filter_options)
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
stoppings = defaultdict(list)  # dict of lists, key is the problem index
timings = defaultdict(list)  # key is the dimension

### go 
print('*** benchmarking %s from %s on suite %s ***'
      % (fmin.__name__, fmin.__module__, suite_name))
time0 = time.time()
for batch_counter, problem in enumerate(suite):  # this loop may take hours or days...
    if batch_counter % batches != current_batch % batches:
        continue
    if not len(timings[problem.dimension]) and len(timings) > 1:
        print("\n   %s %d-D done in %.1e seconds/evaluations"
              % (minimal_print.stime, sorted(timings)[-2],
                 np.median(timings[sorted(timings)[-2]])), end='')
    problem.observe_with(observer)  # generate the data for cocopp post-processing
    problem(np.zeros(problem.dimension))  # making algorithms more comparable
    propose_x0 = problem.initial_solution_proposal  # callable, all zeros in first call
    evalsleft = lambda: int(problem.dimension * budget_multiplier + 1 -
                            max((problem.evaluations, problem.evaluations_constraints)))
    time1 = time.time()
    # apply restarts
    irestart = -1
    while evalsleft() > 0 and not problem.final_target_hit:
        irestart += 1

        # here we assume that `fmin` evaluates the final/returned solution
        if 11 < 3:  # add solver to investigate here
            pass
        elif fmin is scipy.optimize.fmin:
            output = fmin(problem, propose_x0(), maxfun=evalsleft(), disp=False, full_output=True)
            stoppings[problem.index].append(output[4])
        elif fmin is mocmaes.MOCMAES:
            output = mocmaes.MOCMAES.run(problem, propose_x0())
        elif fmin is EGO:
            #
            # # sampling plan
            # criterion = 'EI'  # 'EI' or 'SBO' or 'LCB'
            #
            # # number of points in the initial DOE
            # ndoe = 2  # (at least ndim+1)
            #
            # # number of iterations with EGO
            # n_iter = 5
            #
            # # Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
            #
            # xlimits = np.array([[0, 2]])
            # sampling = LHS(xlimits=xlimits, random_state=1)
            # xdoe = sampling(ndoe)
            #
            # # EGO call
            # ego = EGO(n_iter=n_iter, criterion=criterion, n_doe=ndoe, xlimits=xlimits)
            #
            # x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=problem)
            #
            # output = x_opt

            # k = 2
            # n = 5 * 2
            #
            # X = lhs(2, samples=propose_x0().size)
            # y = np.zeros((propose_x0().size, 2))
            #
            # for i in range(k):
            #     y[i] = problem(X[i])
            #
            # kr = kriging(k, X, y)
            #
            # kr.train()
            #
            #
            # E = EGO(kr)
            # MinExpImp = 1e14
            # infill = 0
            #
            # while infill < 3 * n:
            #     Xnew, EI = E.next_infill()
            #     Ynew = problem(Xnew)
            #     kr.X = np.vstack((kr.X, Xnew))
            #     kr.y = np.vstack((kr.y, Ynew.T))
            #     infill = infill + 1
            #
            #     kr.train()
            #     output = Xnew

            # seed = 666
            # np.random.seed(seed)
            # dim_r = 2  # dimension of the real values
            #
            #
            # # def obj_fun(x):
            # #     x_r = np.array([x['continuous_%d' % i] for i in range(dim_r)])
            # #     x_i = x['ordinal']
            # #     x_d = x['nominal']
            # #     _ = 0 if x_d == 'OK' else 1
            # #     return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + _ * 2
            #
            #
            # # Continuous variables can be specified as follows:
            # # a 2-D variable in [-5, 5]^2
            # # for 2 variables, the naming scheme is continuous0, continuous1
            # C = ContinuousSpace([-5, 5], var_name='continuous') * dim_r
            #
            # # Integer (ordinal) variables can be specified as follows:
            # # The domain of integer variables can be given as with continuous ones
            # # var_name is optional
            # I = OrdinalSpace([5, 15], var_name='ordinal')
            #
            # # Discrete (nominal) variables can be specified as follows:
            # # No lb, ub... a list of categories instead
            # N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], var_name='nominal')
            #
            # # The whole search space can be constructed:
            # search_space = C + I + N
            #
            # # Bayesian optimization also uses a Surrogate model
            # # For mixed variable type, the random forest is typically used
            # model = RandomForest(levels=search_space.levels)
            #
            # # opt = ParallelBO(
            # #     search_space=search_space,
            # #     obj_fun=problem,
            # #     model=model,
            # #     max_FEs=50,
            # #     DoE_size=3,  # the initial DoE size
            # #     eval_type='dict',
            # #     acquisition_fun='MGFI',
            # #     acquisition_par={'t': 2},
            # #     n_job=3,  # number of processes
            # #     n_point=3,  # number of the candidate solution proposed in each iteration
            # #     verbose=True  # turn this off, if you prefer no output
            # # )
            #
            # bo = BO(
            #     search_space=search_space,
            #     obj_fun=problem,
            #     model=model,
            #     eval_type='list',
            #     DoE_size=10,
            #     n_point=1,
            #     acquisition_fun='EI',
            #     verbose=True,
            #     minimize=True
            # )
            #
            # xopt, fopt, stop_dict = bo.run()
            #
            # print('xopt: {}'.format(xopt))
            # print('fopt: {}'.format(fopt))
            # print('stop criteria: {}'.format(stop_dict))
            # output = xopt

            np.random.seed(42)

            dim = 2
            max_FEs = 100
            obj_fun = problem
            lb, ub = -600, 600

            search_space = ContinuousSpace([lb, ub]) * dim
            mean = constant_trend(dim, beta=None)

            # autocorrelation parameters of GPR
            thetaL = 1e-10 * (ub - lb) * np.ones(dim)
            thetaU = 10 * (ub - lb) * np.ones(dim)
            theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

            model = GaussianProcess.GaussianProcess(
                corr='squared_exponential',
                theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                nugget=1e-6, noise_estim=False,
                optimizer='BFGS', wait_iter=5, random_start=5 * dim,
                eval_budget=100 * dim, mean=mean
            )

            bo = BO(
                search_space=search_space,
                obj_fun=problem,
                model=model,
                eval_type='list',
                DoE_size=10,
                n_point=1,
                acquisition_fun='EI',
                verbose=True,
                minimize=True
            )
            xopt, fopt, stop_dict = bo.run()

            print('xopt: {}'.format(xopt))
            print('fopt: {}'.format(fopt))
            print('stop criteria: {}'.format(stop_dict))
            output = xopt

        elif fmin is scipy.optimize.fmin_slsqp:
            output = fmin(problem, propose_x0(), iter=int(evalsleft() / problem.dimension + 1),  # very approximate way to respect budget
                          full_output=True, iprint = -1)
            # print(problem.dimension, problem.evaluations)
            stoppings[problem.index].append(output[3:])
        elif fmin in (cocoex.solvers.random_search, random_search):
            fmin(problem, problem.lower_bounds, problem.upper_bounds, evalsleft())
        elif fmin.__name__ == 'fmin2' and 'cma' in fmin.__module__:  # cma.fmin2:
            xopt, es = fmin(problem, propose_x0, 2,
                            {'maxfevals':evalsleft(), 'verbose':-9}, restarts=9)
            stoppings[problem.index].append(es.stop())
        elif fmin is scipy.optimize.fmin_cobyla:
            fmin(problem, propose_x0(), lambda x: -problem.constraint(x), maxfun=evalsleft(),
                 disp=0, rhoend=1e-9)

    timings[problem.dimension].append((time.time() - time1) / problem.evaluations
                                      if problem.evaluations else 0)
    minimal_print(problem, restarted=irestart, final=problem.index == len(suite) - 1)
    with open(output_folder + '_stopping_conditions.pydict', 'wt') as file_:
        file_.write("# code to read in these data:\n"
                    "# import ast\n"
                    "# with open('%s_stopping_conditions.pydict', 'rt') as file_:\n"
                    "#     stoppings = ast.literal_eval(file_.read())\n"
                    % output_folder)
        file_.write(repr(dict(stoppings)))

### print timings and final message
print("\n   %s %d-D done in %.1e seconds/evaluations"
      % (minimal_print.stime, sorted(timings)[-1], np.median(timings[sorted(timings)[-1]])))
if batches > 1:
    print("*** Batch %d of %d batches finished in %s."
          " Make sure to run *all* batches (via current_batch or batch=#/#) ***"
          % (current_batch, batches, cocoex.utilities.ascetime(time.time() - time0)))
else:
    print("*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))

print("Timing summary:\n"
      "  dimension  median seconds/evaluations\n"
      "  -------------------------------------")
for dimension in sorted(timings):
    print("    %3d       %.1e" % (dimension, np.median(timings[dimension])))
print("  -------------------------------------")

### post-process data
if batches == 1 and 'cocopp' in globals() and cocopp not in (None, 'None'):
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
