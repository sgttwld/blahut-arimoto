import numpy as np
import scipy.stats

def utility(N,K,sigma):
	util = np.zeros((N,K))
	for n in range(0,N):
		for k in range(0,K):
			x = np.arange(0,K,1)
			#util[n,:] = gauss(range(0,K),n,sigma)
			util[n,:] = scipy.stats.norm.pdf(x, loc=n, scale=sigma)
	return util/np.max(util)

def gaussn(x):
	return np.exp(-x**2/2)/np.sqrt(2*np.pi)

def gauss(x,mu,sigma):
	return gaussn((x-mu)/sigma)/sigma

def utility_par(N,K,num=3,sigma1=5.0,sigma2=1.5,shift=6,weight=0.1):
	step = int(K/num)
	U = np.zeros((N,K))
	mu = np.zeros(N)
	nu = np.zeros(N)
	for i in range(0,N):
	    for t in range(0,num):
	        if i < step*(t+1):
	            mu[i] = int(step/2)+t*step
	            if i < step*(t+1)-N/(num*2):
	                nu[i] = mu[i] - shift
	            else:
	                nu[i] = mu[i] + shift
	            break
	    U[i,:] = gauss(range(0,K),mu[i],sigma1) + weight*gauss(range(0,K),nu[i],sigma2)
	return U/np.max(U)

def meye(N,K,a,s):
    """
    modified np.eye matrix allowing for tilted diagonal
    :param a: start i index
    :param s: j steps
    """
    M = np.zeros((N,K))
    for i in range(0,N):
        for j in range(0,K):
            if j>=a and j<a+int(np.ceil(float(N)/s)) and i>=s*(j-a) and i<s*(1+(j-a)):
                M[i,j] = 1
    return M

def meye_sum(N,K,starts,steps,vals=[]):
    """
    weighted sum of meye matrices
    :param starts: list of start j values for the tilted lines
    :param steps: list of i steps defining the tilt of the lines
    """
    if len(vals) == 0:
        vals = [1]*len(steps)
    U = 0
    for a,s,v in zip(starts,steps,vals):
        U += v * meye(N,K,a,s)
    return U
