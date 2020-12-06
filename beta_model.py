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
        return phi_x
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

def fx_pt_convergence_graph(n, num_iter):
    '''
    Creates convergence graphs for the fixed point iteration.
    Args:
        n (int): the size of the network to be created.
        num_iter (int): how many iterations should be carried out.
    '''
    beta = generate_beta(n)
    adj_matrix = generate_graph(beta)
    phi = get_phi(adj_matrix)
    l2_er = np.empty(num_iter)
    l_inf_er = np.empty(num_iter)
    x = np.zeros(n) # Initial Guess
    new_x = phi(x)
    
    # Data Collection
    for i in range(num_iter):
        x = new_x
        new_x = phi(x)
        l2_er[i] = np.linalg.norm(x-new_x)
        l_inf_er[i] = np.linalg.norm(x-new_x, ord=np.inf)

    # Graphical
    plt.clf()
    plt.scatter(np.arange(num_iter), l2_er, label='L^2 Error')
    plt.xlabel('Number of Iterations')
    plt.ylabel('L^2 Error ||x_k-x_{k+1}||_2')
    plt.title(f'Convergence to Fixed Point (L^2 Norm) n={n}')
    plt.savefig('Fixed_Point_Convergence_L2.png')
    plt.clf()
    plt.scatter(np.arange(num_iter), l_inf_er, label='L^infinity Error')
    plt.xlabel('Number of Iterations')
    plt.ylabel('L^infinity Error ||x_k-x_{k+1}||_infinity')
    plt.title(f'Convergence to Fixed Point (Inifinity Norm) n={n}')
    plt.savefig('Fixed_Point_Convergence_L_infinity.png')

def n_convergence_graph(n_max = 500, step = 5, tresh=0.001, executions=5):
    '''
    Creates convergence graphs based on the size of the network.
    Args:
        n_max (int): the maximum size of the network.
        step (int): the starting size of the network and by how much it
            increases for every point.
        tresh (float): the treshold for convergence of fixed point iteration.
            When ||x-phi(x)||_infty < tresh, the algorithm will stop for that
            graph.
        executions (int): How many executions should be carried out for each n.
    '''
    num_points = n_max//step
    l2_er = np.empty(num_points)
    l_inf_er = np.empty(num_points)
    for k in range(num_points):
        n = (k+1)*step
        print(f'\nStarting n = {n}')
        total_l2_er = 0
        total_l_inf_er = 0
        for i in range(executions):
            beta = generate_beta(n)
            adj_matrix = generate_graph(beta)
            phi=get_phi(adj_matrix)
            x = np.zeros(n)
            new_x = phi(x)
            while np.linalg.norm(x-new_x, ord=np.inf) > tresh:
                x = new_x
                new_x = phi(x)
            total_l2_er += np.linalg.norm(new_x-beta)
            total_l_inf_er += np.linalg.norm(new_x-beta, ord=np.inf)
            print(f'Execution #{i+1} out of {executions} done')
        l2_er[k] = total_l2_er/executions
        l_inf_er[k] = total_l_inf_er/executions

    # Graphical
    plt.clf()
    plt.scatter(step*(np.arange(num_points)+1), l2_er, label='L^2 Error')
    plt.xlabel('Size of Network')
    plt.ylabel('L^2 Error ||beta-hat-beta||_2')
    plt.title('Convergence of Estimator (L^2 Norm)')
    plt.savefig('Estimator_Convergence_L2.png')
    plt.clf()
    plt.scatter(step*(np.arange(num_points)+1), l_inf_er, label='L^infinity Error')
    plt.xlabel('Size of Network')
    plt.ylabel('L^infinity Error ||beta-hat-beta||_infinity')
    plt.title('Convergence of Estimator (Infinity Norm)')
    plt.savefig('Estimator_Convergence_L_inf.png')

    return l2_er, l_inf_er


