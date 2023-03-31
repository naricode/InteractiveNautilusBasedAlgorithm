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
import time

''' Initialization Phase:
    Here the weights, the nadir point and the initial nondominated
     solution P are generated. In addition, the problem is defined'''

############### ------------ Defining the problem ------------- ##################
''' River Pollution Problem'''
#---defining the problem parameters----#

problem = RiverPollution()
znadir = [-4.07, -2.80, -0.32, 9.71] #Nadir point of Riverpollution problem
filename = problem.name + "_" + str(problem.num_of_objectives)

#################### Ideal Point #####################
# Minimizing f1 and f2,f3,f4 constraints
z_ideal=[]
fun1 = lambda x: -1*(4.07 + 2.27 * x[0])

cons234= ({'type': 'ineq', 'fun': lambda x: (2.6+ 0.03* x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2))+znadir[1]},
          {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2))+znadir[2]},
          {'type': 'ineq', 'fun': lambda x: -1*(0.96 * (1 / (1.09 - x[1] ** 2) - 1))+znadir[3]})

bnds = ((.5, 1), (.5, 1))

se1=minimize(fun1, (.8, .5), method='SLSQP', bounds=bnds, constraints=cons234)
z_ideal.append(se1.fun)

# Minimizing f2 and f1,f3,f4 constraints
fun2 = lambda x: -1*(2.6+ 0.03* x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2))

cons134= ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0])+znadir[0]},
          {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2))+znadir[2]},
          {'type': 'ineq', 'fun': lambda x: -1*(0.96 * (1 / (1.09 - x[1] ** 2) - 1))+znadir[3]})

se2=minimize(fun2, (.5, .5), method='SLSQP', bounds=bnds, constraints=cons134)
z_ideal.append(se2.fun)

# Minimizing f3 and f1,f2,f4 constraints
fun3 = lambda x: -1 * (8.21 - 0.71 / (1.09 - x[0] ** 2))

cons124= ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0])+znadir[0]},
          {'type': 'ineq', 'fun': lambda x: (2.6+ 0.03* x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2))+znadir[1]},
          {'type': 'ineq', 'fun': lambda x: -1*(0.96 * (1 / (1.09 - x[1] ** 2) - 1))+znadir[3]})

se3=minimize(fun3, (.6, .8), method='SLSQP', bounds=bnds, constraints=cons124)
z_ideal.append(se3.fun)

# Minimizing f4 and f1,f2,f3 constraints
fun4 = lambda x: (0.96 * (1 / (1.09 - x[1] ** 2) - 1))

cons123= ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0]) + znadir[0]},
          {'type': 'ineq', 'fun': lambda x: (2.6+ 0.03* x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2)) + znadir[1]},
          {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2))+znadir[2]})

se4=minimize(fun4, (.6, .8), method='SLSQP', bounds=bnds, constraints=cons123)
z_ideal.append(se4.fun)

#----- Generating Uniform weights W ----#.
lattice_resolution = 15
Wi=Referencepoints(lattice_resolution, problem.num_of_objectives)

## Nµ Representative weights of objectives
Nµ= 500   # The number of weights
NS = 5    # Number of Solutions the DM wants to see in each iteration
W=KMeans(Nµ).fit(Wi.values).cluster_centers_ # Selecting Nµ weights

############## -------------Initial Population ----------------- #####################
pop_init = Population(problem)
individuals1=pop_init.individuals  # Initial individuals (chromososmes)
objectives1=pop_init.objectives    # The objective values of the initial individuals
pop_size=len(individuals1)         # Population size

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
Generations = 20        # Number of generations in WASF-GA to get the initial well-spread set of solutions

for iter in range(NI-1):
############### ------------- Start generating solutions for AB-NAUTILUS ----------------------- ################
     for gen in range(Generations):
          #-------------Crossover & Mutation----------------#
          print("# WASF-GA_iteration",gen, "in NI",iter+1)
          """Conduct crossover and mutation over the population.
          Conduct simulated binary crossover and bounded polunomial mutation.          """

          offspring = pop_init.mate(individuals1) # Producing the children (crossover and mutation)
          obj_offs=np.empty((0, problem.num_of_objectives), float) # Evaluate the children (the offspring's objectives values)

          #-------------Runnig in the background----------------#
          """ Creating and updating the individuals in the background"""

          if __name__ == '__main__':
               x = Thread(target=pop_init.mate, args=(individuals1,))
               x.daemon=True
               x.start()
               
          for i in range(len(offspring)):
               obj_offs = np.vstack((obj_offs,problem.objectives(offspring[i])))

          #-----------------Combinations to create the population with size 2N--------------------#
          individuals=np.vstack((individuals1, offspring))
          objectives=np.vstack((objectives1, obj_offs))

          #---------------------Selection Process using ASF-------------------------#

          ##-------ASF Evaluation--------##
          ''' For each weight, evaluate the ASF for all solutions,
          the solution with the minimum ASF is chosen for each weight'''

          objectives1=np.empty((0, problem.num_of_objectives), float)      # Update the objective values based on new individuals
          individuals1= np.empty((0, problem.num_of_variables), float)     # Update the individuals
          K1=np.zeros((len(W)))
          K2=np.zeros((len(W)))
          for j in range(len(W)):
               weight=W[j] 
               asf=list()
               
               for i in range(len(objectives)):
                    obj=objectives[i]  
                    asf_val=ASF(problem, obj, weight, reference_p).asf_fun()
                    asf.append(asf_val)

               asf_sort=sorted(asf)     # sort the ASF values ascending
               K=np.argsort(asf)        # The index ordered ascending based on sorted values
               K1[j]=K[0]               # The index of the solution with the minimal ASF value
               K2[j]=asf_sort[0]        # The ASF value related to the K1 index
          
          no_redun_sol = list(dict.fromkeys(K1)) # Remove the redundant solutions (the solutions chosen several times for different weights) from the list K1

          # Select pop_size solutions for next generation
          if len(no_redun_sol) > pop_size:
               no_redun_sol=no_redun_sol[0:pop_size]

          ## Sorting the individuals and original objective functions based on their ASF values
          for i in range(pop_size):
               objectives1 = np.vstack((objectives1, objectives[int(no_redun_sol[i])]))
               individuals1 = np.vstack((individuals1, individuals[int(no_redun_sol[i])]))

          #if gen==0:
               #figure = animate_init_(objectives1, filename + ".html") #revised in 2023.03.23
               #figure= animate_next_(objectives1, figure, filename + ".html",gen)
          #else:
               #figure= animate_next_(objectives1, figure, filename + ".html",gen) #revised in 2023.03.23
               # if gen==20

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
     dis_to_NsPoints=[]
     i=0
     for point_idx in closest:
          NsPoints.append(objectives1[point_idx]) # the most representative solutions
          dis_to_NsPoints.append(np.linalg.norm(reference_p-NsPoints[i])) # the Euclidean distance from the reference_point to the representative points NSPoints
          i=i+1
     
     zh_prev = reference_p # the previous reference point (reference_p) is set as zh_prev
     interm_points=np.empty((0, problem.num_of_objectives), float)

     ### Calculating the intermediate points ###
     for i in range(len(NsPoints)):
          interm_points=np.vstack((interm_points, [(((ith-1)/ith)*zh_prev[j])+((1/ith)*NsPoints[i][j]) for j in range(problem.num_of_objectives)]))
     
     intermed_P=np.vstack((intermed_P, interm_points))
     # The DM selects a point (here reference_point is selected randomly)
     
     index = random.randrange(len(interm_points)) # the index of selected point randomly
     index=index-1
     reference_p = interm_points[index] # the reference_p is updated by the seleted point in the current iteration
     LRP=np.vstack((LRP,reference_p)) # update the list of previous points

     #####----------Calculating the lower bound z_lo-----------######
     z_lo=[]

     # Minimizing f1 and f2,f3,f4 constraints
     fun1 = lambda x: -1*(4.07 + 2.27 * x[0])
     
     cons234= ({'type': 'ineq', 'fun': lambda x: (2.6+ 0.03* x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2))+reference_p[1]},
            {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2))+reference_p[2]},
            {'type': 'ineq', 'fun': lambda x: -1*(0.96 * (1 / (1.09 - x[1] ** 2) - 1))+reference_p[3]})

     bnds = ((.5, 1), (.5, 1))

     se1=minimize(fun1, (.8, .5), method='SLSQP', bounds=bnds, constraints=cons234)
     z_lo.append(se1.fun)

     # Minimizing f2 and f1,f3,f4 constraints
     fun2 = lambda x: -1*(2.6+ 0.03* x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2))
     
     cons134= ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0])+reference_p[0]},
            {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2))+reference_p[2]},
            {'type': 'ineq', 'fun': lambda x: -1*(0.96 * (1 / (1.09 - x[1] ** 2) - 1))+reference_p[3]})

     se2=minimize(fun2, (.5, .5), method='SLSQP', bounds=bnds, constraints=cons134)
     z_lo.append(se2.fun)

     # Minimizing f3 and f1,f2,f4 constraints
     fun3 = lambda x: -1 * (8.21 - 0.71 / (1.09 - x[0] ** 2))
     
     cons124= ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0])+reference_p[0]},
            {'type': 'ineq', 'fun': lambda x: (2.6+ 0.03* x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2))+reference_p[1]},
            {'type': 'ineq', 'fun': lambda x: -1*(0.96 * (1 / (1.09 - x[1] ** 2) - 1))+reference_p[3]})

     se3=minimize(fun3, (.6, .8), method='SLSQP', bounds=bnds, constraints=cons124)
     z_lo.append(se3.fun)

     # Minimizing f4 and f1,f2,f3 constraints
     fun4 = lambda x: (0.96 * (1 / (1.09 - x[1] ** 2) - 1))
     
     cons123= ({'type': 'ineq', 'fun': lambda x: (4.07 + 2.27 * x[0]) + reference_p[0]},
            {'type': 'ineq', 'fun': lambda x: (2.6+ 0.03* x[0] + 0.02 * x[1] + 0.01 / (1.39 - x[0] ** 2) + 0.3 / (1.39 - x[1] ** 2)) + reference_p[1]},
            {'type': 'ineq', 'fun': lambda x: (8.21 - 0.71 / (1.09 - x[0] ** 2))+reference_p[2]})

     se4=minimize(fun4, (.6, .8), method='SLSQP', bounds=bnds, constraints=cons123)
     z_lo.append(se4.fun)

     z_low=np.vstack((z_low,z_lo))
     # Agent stores and updates the solutions in the current iteration
     P_new=np.empty((0, problem.num_of_objectives), float)
     indiv_new=np.empty((0, problem.num_of_variables), float)
     for i in range(pop_size):
          Pupd_curr_iter=[]
          indiv_curr_iter=[]
          for j in range(problem.num_of_objectives):
               if objectives1[i][j] >= z_lo[j] and objectives1[i][j] <= reference_p[j]: 
                    Pupd_curr_iter.append(objectives1[i][j])
               
          if len(Pupd_curr_iter)==problem.num_of_objectives:
               P_new=np.vstack((P_new,Pupd_curr_iter)) # P_new is the objective values of the set of nondominated solutions are in the current reachable area
               indiv_curr_iter.append(individuals1[i])
               indiv_new=np.vstack((indiv_new,indiv_curr_iter))  # indiv_new is the set of nondominated solutions updated based on the current reachable area

     objectives1=P_new
     individuals1=indiv_new
     pop_size=len(individuals1)
     #if ith==1
        #pre_proc_individ=

     ith=ith-1 # Updating the number of iterations left to the Pareto solutions

print("LRP: ", LRP)
print("z_low: ", z_low)

interm_points=np.vstack((interm_points,z_ideal, znadir)) # The intermediate points stacked with nadir and ideal points

 # The graphical representation of intermediate points stacked with nadir and ideal points
figure = animate_init_(intermed_P, filename + ".html")

LRP=np.vstack((LRP,z_ideal)) # List of reference points stacked with ideal point
figure = animate_init_(LRP, filename + ".html") # Graphical representation of reference point