import numpy as np
import matplotlib.pyplot as plt

def generate_beta(n, low=-10, high=10):
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
    adj_matix = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            factor = np.exp(beta[i]+beta[j])
            if np.random.random() < factor/(1+factor):
                adj_matrix[i,j] = 1
                adj_matrix[j,i] = 1
    return adj_matrix


