import numpy as np

def gaussn(x):
	return np.exp(-x**2/2)/np.sqrt(2*np.pi)

def gauss(x,mu,sigma):
	return gaussn((x-mu)/sigma)/sigma

def utility(N,K,sigma):
	util = np.zeros((N,K))
	for n in range(0,N):
		for k in range(0,K):
			x = np.arange(0,K,1)
			util[n,:] = gauss(range(0,K),n,sigma)
	return util/np.max(util)
