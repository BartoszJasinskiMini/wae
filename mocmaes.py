import numpy as np
import matplotlib.pyplot as plt
import math
import sys

mu = 100	
maxGenerations = 50	

pTarget = pow(5 + math.sqrt(0.5), -1) 
dDamping = 1 + mu/2 			
cSuccRateParam = pTarget/(2 + pTarget)	
cCumulTimeParam = 2/(2 + mu)		
cCov = 2/(pow(mu,2) + 6)		
pThresh = 0.44	

MAX = pow(10,6)

def computeCrowdingDistances(individuals, n, dim):

	crowdingDistances = np.zeros(n)	
	dim_i = np.zeros(n)			

	for i in range(dim):
		for j in range(n):
			dim_i[j] = individuals[j].fitness[i]

		perm = dim_i.argsort()
		dim_i = dim_i[perm]

		crowdingDistances[perm[0]] = MAX
		crowdingDistances[perm[n-1]] = MAX

		for j in range(2, n-1):
			crowdingDistances[perm[j]] += (dim_i[i+1] - dim_i[i-1]) / (dim_i[n-1] - dim_i[0])

	return crowdingDistances

def selectBest(individuals, mu):

	n = len(individuals)			
	dim = individuals[0].fitness.size	

	crowdingDistances = computeCrowdingDistances(individuals, n, dim)
	
	nonDominationRanks = np.zeros(n)

	for i in range(n):				
		for j in range(n):
			if individuals[i].dominates(individuals[j]):
				nonDominationRanks[j] += 1

	perm = nonDominationRanks.argsort()
	nonDominationRanks = nonDominationRanks[perm]	
	individuals = np.array(individuals)[perm]

	nextGenCount = 0			# keeps count of the number of individuals in the next generation
	nextGen = []				# stores the best mu individuals and builds the next generation
	
	for level in range(2 * mu):		# for each possible level of nondominance: sort all individuals of..
		tempIndividuals = []		# ..that level according to their crowding distance
		tempCrowdingDistances = []
		while(nonDominationRanks[nextGenCount] == level and nextGenCount < mu):
			tempIndividuals.append(individuals[nextGenCount])
			tempCrowdingDistances.append(crowdingDistances[nextGenCount])
			nextGenCount += 1

		if (len(tempIndividuals) > 0):
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
		result = False
		for i in range(self.outputDim):
			if self.fitness[i] > other.fitness[i]:		
				return False
			if self.fitness[i] < other.fitness[i]:		
				result = True
		return result

	def updateStepSize(self):
		if self.succ:
			self.pSucc = (1 - cSuccRateParam)*self.pSucc + cSuccRateParam
		else:
			self.pSucc = (1 - cSuccRateParam)*self.pSucc

		self.sigma = self.sigma * math.exp((self.pSucc - pTarget) / (dDamping*(1 - pTarget)))
		
	def updateCovariance(self):
		if self.pSucc < pThresh:
			self.pEvol = (1 - cCumulTimeParam) * self.pEvol + math.sqrt(cCumulTimeParam * (2 - cCumulTimeParam)) * self.step
			self.C = (1 - cCov) * self.C + cCov * (np.transpose(self.pEvol) * self.pEvol)
		else:
			self.pEvol = (1 - cCumulTimeParam)*self.pEvol
			self.C = (1 - cCov)*self.C + cCov*(np.transpose(self.pEvol) * self.pEvol + cCumulTimeParam * (2 - cCumulTimeParam)*self.C)

	def mutate(self):
		newx = np.random.multivariate_normal(self.x, pow(self.sigma,2) * self.C)

		mutation = Individual(newx, self.pSucc, self.sigma, self.pEvol, self.C, self.f)

		mutation.step = (mutation.x - self.x)/self.sigma
		mutation.succ = mutation.dominates(self)
		self.succ = mutation.succ

		return mutation

class MOCMAES:

	def run(func, x0):		
			initialSigma = 5.0				
			inputDim = x0.size

			currentPop = []	
			for i in range(mu):
				xi = np.random.rand(inputDim) * 4 * initialSigma - 2 * initialSigma
				Ci = np.identity(inputDim)
				currentPop.append(Individual(xi, pTarget, initialSigma, 0, Ci, func))

			for g in range(0, maxGenerations):

				Q = []
				for k in range(mu):
					Q.append(currentPop[k].mutate())

				for k in range(mu):
					currentPop[k].updateStepSize()
					Q[k].updateStepSize()

					Q[k].updateCovariance()

					Q.append(currentPop[k])

				currentPop = selectBest(Q, mu)

			return currentPop[0].x

