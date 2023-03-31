#If you use any part of this research in your own work, please cite the following reference using APA 7 format:
# [insert reference].
#This project has been published in the Journal of Business Analytics, which is available here:
# [insert link].
# Please note that this code requires Python 3.7 to run, and the functionality for creating figures and saving files has been disabled when comparing algorithms."

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.optimize import fsolve, minimize
from sklearn.cluster import KMeans
from weights import Referencepoints
from river_pollution import RiverPollution
from Population import Population
from ASF import ASF
import random
from random import shuffle
from pyrvea.OtherTools.plotlyanimate import animate_init_, animate_next_
from desdeo.core.ResultFactory import BoundsFactory, IterationPointFactory
from desdeo.optimization.OptimizationMethod import OptimizationMethod
from threading import Thread
from multiprocessing import Pool, Process, Queue
# https://pymoo.org/algorithms/moo/nsga3.html
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from Problem_RiverPollution_SetForNSGA3 import RiverPollution_SetForNSGA3
import time

''' Initialization Phase:
    Here the weights, the nadir point and the initial nondominated
     solution P are generated. In addition, the problem is defined'''

############### ------------ Defining the problem ------------- ##################
''' River Pollution Problem'''
# ---defining the problem parameters----#

problem = RiverPollution()
znadir = [-4.07, -2.80, -0.32, 9.71]  # Nadir point of Riverpollution problem
filename = problem.name + "_" + str(problem.num_of_objectives)

#################### Ideal Point #####################
# Minimizing f1 and f2,f3,f4 constraints
z_ideal = []
fun1 = lambda x: -1 * (4.07 + 2.27 * x[0])

cons234 = ({'type': 'ineq',
            'fun': lambda x: (2.6 + 0.03 * x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2)) +
                             znadir[1]},
           {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2)) + znadir[2]},
           {'type': 'ineq', 'fun': lambda x: -1 * (0.96 * (1 / (1.09 - x[1] ** 2) - 1)) + znadir[3]})

bnds = ((.5, 1), (.5, 1))

se1 = minimize(fun1, (.8, .5), method='SLSQP', bounds=bnds, constraints=cons234)
z_ideal.append(se1.fun)

# Minimizing f2 and f1,f3,f4 constraints
fun2 = lambda x: -1 * (2.6 + 0.03 * x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2))

cons134 = ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0]) + znadir[0]},
           {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2)) + znadir[2]},
           {'type': 'ineq', 'fun': lambda x: -1 * (0.96 * (1 / (1.09 - x[1] ** 2) - 1)) + znadir[3]})

se2 = minimize(fun2, (.5, .5), method='SLSQP', bounds=bnds, constraints=cons134)
z_ideal.append(se2.fun)

# Minimizing f3 and f1,f2,f4 constraints
fun3 = lambda x: -1 * (8.21 - 0.71 / (1.09 - x[0] ** 2))

cons124 = ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0]) + znadir[0]},
           {'type': 'ineq',
            'fun': lambda x: (2.6 + 0.03 * x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2)) +
                             znadir[1]},
           {'type': 'ineq', 'fun': lambda x: -1 * (0.96 * (1 / (1.09 - x[1] ** 2) - 1)) + znadir[3]})

se3 = minimize(fun3, (.6, .8), method='SLSQP', bounds=bnds, constraints=cons124)
z_ideal.append(se3.fun)

# Minimizing f4 and f1,f2,f3 constraints
fun4 = lambda x: (0.96 * (1 / (1.09 - x[1] ** 2) - 1))

cons123 = ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0]) + znadir[0]},
           {'type': 'ineq',
            'fun': lambda x: (2.6 + 0.03 * x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2)) +
                             znadir[1]},
           {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2)) + znadir[2]})

se4 = minimize(fun4, (.6, .8), method='SLSQP', bounds=bnds, constraints=cons123)
z_ideal.append(se4.fun)

# ----- Generating Uniform weights W ----#.
lattice_resolution = 15
Wi = Referencepoints(lattice_resolution, problem.num_of_objectives)

## Nµ Representative weights of objectives
Nµ = 500  # The number of weights
NS = 5  # Number of Solutions the DM wants to see in each iteration
W = KMeans(Nµ).fit(Wi.values).cluster_centers_  # Selecting Nµ weights

############## -------------Initial Population ----------------- #####################
pop_init = Population(problem)
individuals1 = pop_init.individuals  # Initial individuals (chromososmes)
objectives1 = pop_init.objectives  # The objective values of the initial individuals
pop_size = len(individuals1)  # Population size

''' Interactive Phase:
     the DM provides the preferences, and the agent is used
     to handle the populations intelligently'''

############# ------------- Loop for Interactive process (number of iteration NI) --------------- #############

NI = 3                   # Number of iterations
ith=NI                   # Number of iterations remained to reach the representative solutions which is equal to NI initially
reference_p = znadir     # reference point in WASF-GA which can be any value (here equal to nadir point)
LRP=znadir               # List of previous reference points
z_low=z_ideal
intermed_P=np.empty((0, problem.num_of_objectives), float)
rho=0.1                  # Parameter in ASF
Generations = 100         # Number of generations in WASF-GA to get the initial well-spread set of solutions

for iter in range(NI - 1):
    ############### ------------- Start generating solutions for AB-NAUTILUS ----------------------- ################
    from pymoo.optimize import minimize
    problem4 = RiverPollution_SetForNSGA3()
    algorithm3 = NSGA2(pop_size=100,
                      sampling=BinaryRandomSampling(),
                      crossover=TwoPointCrossover(),
                      eliminate_duplicates=True)
    res = minimize(problem4,
                       algorithm3,
                       ('n_gen', 100),
                       seed=1,
                       verbose=False) #verbose=True --> will show the results

    pop = res.pop
    no_redun_sol = res.X
    objectives1 = pop.get("F")

    # k-mean cluster Ns solutions
    k_means = KMeans(NS)
    solution_points = objectives1
    k_means.fit(solution_points)

    # Closest solutions to the center of each class are selected
    closest = set(
        pairwise_distances_argmin_min(
            k_means.cluster_centers_, solution_points
        )[0]
    )
    NsPoints = []
    dis_to_NsPoints = []
    i = 0
    for point_idx in closest:
        NsPoints.append(objectives1[point_idx])  # the most representative solutions
        dis_to_NsPoints.append(np.linalg.norm(reference_p - NsPoints[
            i]))  # the Euclidean distance from the reference_point to the representative points NSPoints
        i = i + 1

    zh_prev = reference_p  # the previous reference point (reference_p) is set as zh_prev
    #interm_points = np.empty((0, problem.num_of_objectives), float)
    interm_points = np.empty((0, 4), float)

    ### Calculating the intermediate points ###
    for i in range(len(NsPoints)):
        interm_points = np.vstack((interm_points,
                                   [(((ith - 1) / ith) * zh_prev[j]) + ((1 / ith) * NsPoints[i][j]) for j in
                                    #range(problem.num_of_objectives)]))
                                    range(4)]))

    intermed_P = np.vstack((intermed_P, interm_points))
    # The DM selects a point (here reference_point is selected randomly)

    index = random.randrange(len(interm_points))  # the index of selected point randomly
    index = index - 1
    reference_p = interm_points[index]  # the reference_p is updated by the seleted point in the current iteration
    LRP = np.vstack((LRP, reference_p))  # update the list of previous points

    #####----------Calculating the lower bound z_lo-----------######
    from scipy.optimize import fsolve, minimize
    z_lo = []
    # Minimizing f1 and f2,f3,f4 constraints
    fun1 = lambda x: -1 * (4.07 + 2.27 * x[0])

    cons234 = ({'type': 'ineq', 'fun': lambda x: (2.6 + 0.03 * x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (
                1.39 - x[1] ** 2)) + reference_p[1]},
               {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2)) + reference_p[2]},
               {'type': 'ineq', 'fun': lambda x: -1 * (0.96 * (1 / (1.09 - x[1] ** 2) - 1)) + reference_p[3]})

    bnds = ((.5, 1), (.5, 1))

    se1 = minimize(fun1, (.8, .5), method="SLSQP", bounds=bnds, constraints=cons234)
    z_lo.append(se1.fun)

    # Minimizing f2 and f1,f3,f4 constraints
    fun2 = lambda x: -1 * (2.6 + 0.03 * x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2))

    cons134 = ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0]) + reference_p[0]},
               {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2)) + reference_p[2]},
               {'type': 'ineq', 'fun': lambda x: -1 * (0.96 * (1 / (1.09 - x[1] ** 2) - 1)) + reference_p[3]})

    se2 = minimize(fun2, (.5, .5), method='SLSQP', bounds=bnds, constraints=cons134)
    z_lo.append(se2.fun)

    # Minimizing f3 and f1,f2,f4 constraints
    fun3 = lambda x: -1 * (8.21 - 0.71 / (1.09 - x[0] ** 2))

    cons124 = ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0]) + reference_p[0]},
               {'type': 'ineq', 'fun': lambda x: (2.6 + 0.03 * x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (
                           1.39 - x[1] ** 2)) + reference_p[1]},
               {'type': 'ineq', 'fun': lambda x: -1 * (0.96 * (1 / (1.09 - x[1] ** 2) - 1)) + reference_p[3]})

    se3 = minimize(fun3, (.6, .8), method='SLSQP', bounds=bnds, constraints=cons124)
    z_lo.append(se3.fun)

    # Minimizing f4 and f1,f2,f3 constraints
    fun4 = lambda x: (0.96 * (1 / (1.09 - x[1] ** 2) - 1))

    cons123 = ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0]) + reference_p[0]},
               {'type': 'ineq', 'fun': lambda x: (2.6 + 0.03 * x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (
                           1.39 - x[1] ** 2)) + reference_p[1]},
               {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2)) + reference_p[2]})

    se4 = minimize(fun4, (.6, .8), method='SLSQP', bounds=bnds, constraints=cons123)
    z_lo.append(se4.fun)

    z_low = np.vstack((z_low, z_lo))
    # Agent stores and updates the solutions in the current iteration
    P_new = np.empty((0, problem.num_of_objectives), float)
    indiv_new = np.empty((0, problem.num_of_variables), float)
    for i in range(pop_size):
        Pupd_curr_iter = []
        indiv_curr_iter = []
        for j in range(problem.num_of_objectives):
            if objectives1[i][j] >= z_lo[j] and objectives1[i][j] <= reference_p[j]:
                Pupd_curr_iter.append(objectives1[i][j])

        if len(Pupd_curr_iter) == problem.num_of_objectives:
            P_new = np.vstack((P_new,
                               Pupd_curr_iter))  # P_new is the objective values of the set of nondominated solutions are in the current reachable area
            indiv_curr_iter.append(individuals1[i])
            indiv_new = np.vstack((indiv_new,
                                   indiv_curr_iter))  # indiv_new is the set of nondominated solutions updated based on the current reachable area

    objectives1 = P_new
    individuals1 = indiv_new
    pop_size = len(individuals1)
    # if ith==1
    # pre_proc_individ=

    #Generations = 20
    ith = ith - 1  # Updating the number of iterations left to the Pareto solutions

print("LRP: ", LRP)
print("z_low: ", z_low)

interm_points = np.vstack(
    (interm_points, z_ideal, znadir))  # The intermediate points stacked with nadir and ideal points

# The graphical representation of intermediate points stacked with nadir and ideal points
#figure = animate_init_(intermed_P, filename + ".html")

LRP = np.vstack((LRP, z_ideal))  # List of reference points stacked with ideal point
figure = animate_init_(LRP, filename + ".html")  # Graphical representation of reference point