import numpy as np
import matplotlib.pyplot as plt
import math
import sys

mu = 100		# population size
maxGenerations = 50	# maximum number of generations

pTarget = pow(5 + math.sqrt(0.5), -1) 	# target success probability
dDamping = 1 + mu/2 			# step size damping parameter
cSuccRateParam = pTarget/(2 + pTarget)	# success rate averaging parameter
cCumulTimeParam = 2/(2 + mu)		# cumulation time horizon parameter
cCov = 2/(pow(mu,2) + 6)		# covariance matrix learning rate
pThresh = 0.44	

MAX = pow(10,2)	# crowding distance that is assigned to min and max points

# computes the crowding distance for each individual in the list
def computeCrowdingDistances(individuals, n, dim):

	crowdingDistances = np.zeros(n)		# stores the crowding distance of each individual
	dimi = np.zeros(n)			# stores fitness values of dimension i

	# for each i sort the fitness values of the individuals by their i-th dimension
	for i in range(dim):
		
		for j in range(n):
			dimi[j] = individuals[j].fitness[i]

		perm = dimi.argsort()
		dimi = dimi[perm]

		# assign MAX crowding distance to min and max points
		crowdingDistances[perm[0]] = MAX
		crowdingDistances[perm[n-1]] = MAX

		# compute crowding distance for all points in between
		for j in range(2,n-1):
			crowdingDistances[perm[j]] += (dimi[i+1] - dimi[i-1]) / (dimi[n-1] - dimi[0])

	# return the result
	return crowdingDistances

def selectBest(individuals, mu):

	n = len(individuals)			# get the number of individuals
	dim = individuals[0].fitness.size	# get the number of objectives

	# compute crowding distance
	crowdingDistances = computeCrowdingDistances(individuals, n, dim)
	
	# assign and sort by nondomination ranks
	nonDominationRanks = np.zeros(n)

	for i in range(n):				# naive O(n^2) implementation
		for j in range(n):
			if individuals[i].dominates(individuals[j]):
				nonDominationRanks[j] += 1

	perm = nonDominationRanks.argsort()
	nonDominationRanks = nonDominationRanks[perm]	# sort nondomination ranks
	individuals = np.array(individuals)[perm]	# sort individuals by nondomination ranks


	# sort each level of nondomination according to the crowding distance
	nextGenCount = 0			# keeps count of the number of individuals in the next generation
	nextGen = []				# stores the best mu individuals and builds the next generation
	
	for level in range(2*mu):		# for each possible level of nondominance: sort all individuals of..
		tempIndividuals = []		# ..that level according to their crowding distance
		tempCrowdingDistances = []
		while(nonDominationRanks[nextGenCount] == level and nextGenCount < mu):
			tempIndividuals.append(individuals[nextGenCount])
			tempCrowdingDistances.append(crowdingDistances[nextGenCount])
			nextGenCount += 1

		if (len(tempIndividuals) > 0):	# if there are no individuals with that level of nondominance: no need to sort
			tempIndividuals = np.array(tempIndividuals)
			tempCrowdingDistances = np.array(tempCrowdingDistances)
			
			perm = tempCrowdingDistances.argsort()
			tempIndividuals = tempIndividuals[perm]
			tempCrowdingDistances = tempCrowdingDistances[perm]

			for indiv in tempIndividuals:
				nextGen.append(indiv)

	return nextGen

class Individual:

	def __init__(self, _x, _pSucc, _sigma, _pEvol, _C, _f):
		self.x = _x
		self.pSucc = _pSucc
		self.sigma = _sigma
		self.pEvol = _pEvol
		self.C = _C
		self.f = _f

		self.fitness = _f(self.x)

		self.inputDim = self.x.size
		self.outputDim = self.fitness.size
		

	def dominates(self, other):
		temp = False
		for i in range(self.outputDim):
			
			# check whether x[i] <= y[i] for all
			if self.fitness[i] > other.fitness[i]:		
				return False
			
			# check whether exists x[i] < y[i]
			if self.fitness[i] < other.fitness[i]:		
				temp = True
		return temp

	def updateStepSize(self):
		if self.succ:
			# if mutation was successful: increase pSucc
			self.pSucc = (1 - cSuccRateParam)*self.pSucc + cSuccRateParam
		else:
			# if mutation was not successful: descrease pSucc
			self.pSucc = (1 - cSuccRateParam)*self.pSucc

		# increase step size if success probability pSucc is bigger than target success probability pTarget
		self.sigma = self.sigma * math.exp((self.pSucc - pTarget) / (dDamping*(1 - pTarget)))
		
	def updateCovariance(self):
		if self.pSucc < pThresh:
			# if the success rate is smaller than the threshold the mutation step is used to update the covariance matrix
			self.pEvol = (1 - cCumulTimeParam)*self.pEvol + math.sqrt(cCumulTimeParam*(2 - cCumulTimeParam))*self.step
			self.C = (1 - cCov)*self.C + cCov*(np.transpose(self.pEvol)*self.pEvol)
		else:
			# if the success rate is higher than the threshold the mutation steop is not used for the update
			self.pEvol = (1 - cCumulTimeParam)*self.pEvol
			self.C = (1 - cCov)*self.C + cCov*(np.transpose(self.pEvol)*self.pEvol + cCumulTimeParam*(2 - cCumulTimeParam)*self.C)

	def mutate(self):

		# find x of the mutation
		newx = np.random.multivariate_normal(self.x, pow(self.sigma,2)*self.C)
		
		# create mutated individual
		mutation = Individual(newx, self.pSucc, self.sigma, self.pEvol, self.C, self.f)
		
		# set mutation step and check whether the mutation dominates its parent
		mutation.step = (mutation.x - self.x)/self.sigma
		mutation.succ = mutation.dominates(self)
		self.succ = mutation.succ

		return mutation

class MOCMAES:

	def run(func, x0):
			
			initialSigma = 5.0				# initial sigma and initial mean determine..
			inputDim = x0.size

			currentPop = []					# create an initial population in initialMean +- 2*initialSigma
			for i in range(mu):
				xi = np.random.rand(inputDim)*4*initialSigma - 2*initialSigma
				Ci = np.identity(inputDim)
				currentPop.append(Individual(xi, pTarget, initialSigma, 0, Ci, func))

			# loop
			for g in range(0, maxGenerations):

				# step 1: reproduction
				Q = []
				for k in range(mu):
					Q.append(currentPop[k].mutate())

				# step 2: updates
				for k in range(mu):
					# update step size
					currentPop[k].updateStepSize()
					Q[k].updateStepSize()

					# update covariance matrix
					Q[k].updateCovariance()

					#create mixed population
					Q.append(currentPop[k])

				# step 3: selection
				currentPop = selectBest(Q, mu)

			return currentPop[0].x

