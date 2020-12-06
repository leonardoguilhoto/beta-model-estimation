import numpy as np
import matplotlib.pyplot as plt

def generate_beta(n, low=-1, high=1):
    '''
    Generates an n-dimensional beta array where components are chosen
    uniformly in the interval [low, high]
    '''
    beta = np.random.random(n)*(high-low) + low
    return beta

def generate_graph(beta):
    '''
    Given a parameter beta, creates a graph according to the
    beta-model. Returns the adjecency matrix of the graph.
    '''
    n = len(beta)
    adj_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            factor = np.exp(beta[i]+beta[j])
            if np.random.random() < factor/(1+factor):
                adj_matrix[i,j] = 1
                adj_matrix[j,i] = 1
    return adj_matrix


def get_phi(adj_matrix):
    '''
    Given an adjecency matrix for a graph in the beta-model, returns
    the function phi of that graph to be used in fixed point
    iteration.
    '''
    n = len(adj_matrix)
    deg_seq = np.zeros(n)
    for i in range(n):
        for j in range(n):
            deg_seq[i] += adj_matrix[i,j]
    def phi(x):
        phi_x = np.empty(n)
        for i in range(n):
            sums = 0
            for j in range(n):
                if i != j:
                    sums += 1/(np.exp(x[i])+np.exp(-x[j]))
            phi_x[i] = np.log(deg_seq[i])-np.log(sums)
        return phi_x #FIXME what if a node has degree 0?
    return phi

def fixed_iter(phi, x_0, n=10**2):
    '''
    Given a function phi and an initial guess x_0, computes n steps
    of fixed iteration and returns the new result.
    '''
    x = x_0
    for _ in range(n):
        x = phi(x)
    return x

