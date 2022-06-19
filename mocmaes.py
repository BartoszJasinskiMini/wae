import numpy as np
import matplotlib.pyplot as plt
import math
import sys

popSize = 100	
maxGenerations = 50	

lambd = 1
dDamping = 1 + popSize/(2 * lambd) 	
pTarget = pow(5 + math.sqrt(lambd / 2.0), -1)
cP = (pTarget * lambd)/(2 + (pTarget * lambd))	
cC = 2/(2 + popSize)		
cCov = 2/(pow(popSize,2) + 6)		
pThresh = 0.44	

INF = pow(10,10)

def computeCrowdingDistances(individuals, n, dim):

	crowdingDistances = np.zeros(n)	
	dim_i = np.zeros(n)			

	for i in range(dim):
		for j in range(n):
			dim_i[j] = individuals[j].fitness[i]

		perm = dim_i.argsort()
		dim_i = dim_i[perm]

		crowdingDistances[perm[0]] = INF
		crowdingDistances[perm[n-1]] = INF

		for j in range(2, n-1):
			crowdingDistances[perm[j]] += (dim_i[i+1] - dim_i[i-1]) / (dim_i[n-1] - dim_i[0])

	return crowdingDistances

def selectBest(individuals, popSize):
	
	nextGen = []
	nextGenCount = 0			
	n = len(individuals)			
	dim = individuals[0].fitness.size	

	crowdingDistances = computeCrowdingDistances(individuals, n, dim)
	
	nonDominationRanks = np.zeros(n)

	for i in range(n):				
		for j in range(n):
			if individuals[i].dominates(individuals[j]):
				nonDominationRanks[j] += 1

	perm = nonDominationRanks.argsort()
	individuals = np.array(individuals)[perm]
	nonDominationRanks = nonDominationRanks[perm]	
	
	for level in range(2 * popSize):		
		tempIndividuals = []
		tempCrowdingDistances = []
		while(nonDominationRanks[nextGenCount] == level and nextGenCount < popSize):
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

	def updateStepSize(self):
		if self.succ:
			self.pSucc = (1 - cP)*self.pSucc + cP
		else:
			self.pSucc = (1 - cP)*self.pSucc

		self.sigma = self.sigma * math.exp((self.pSucc - pTarget) / (dDamping*(1 - pTarget)))
		
	def updateCovariance(self):
		if self.pSucc < pThresh:
			self.pEvol = (1 - cC) * self.pEvol + math.sqrt(cC * (2 - cC)) * self.step
			self.C = (1 - cCov) * self.C + cCov * (np.transpose(self.pEvol) * self.pEvol)
		else:
			self.pEvol = (1 - cC) * self.pEvol
			self.C = (1 - cCov) * self.C + cCov * (np.transpose(self.pEvol) * self.pEvol + cC * (2 - cC)*self.C)

	def dominates(self, other):
		result = False
		for i in range(self.outputDim):
			if self.fitness[i] > other.fitness[i]:		
				return False
			if self.fitness[i] < other.fitness[i]:		
				result = True
		return result

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
			for i in range(popSize):
				xi = np.random.rand(inputDim) * 4 * initialSigma - 2 * initialSigma
				Ci = np.identity(inputDim)
				currentPop.append(Individual(xi, pTarget, initialSigma, 0, Ci, func))

			for g in range(0, maxGenerations):

				Q = []
				for k in range(popSize):
					Q.append(currentPop[k].mutate())

				for k in range(popSize):
					currentPop[k].updateStepSize()
					Q[k].updateStepSize()

					Q[k].updateCovariance()

					Q.append(currentPop[k])

				currentPop = selectBest(Q, popSize)

			return currentPop[0].x

