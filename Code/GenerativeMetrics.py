'''
    Author: Fasil Cheema
    Purpose: Compute Density and Coverage as described in 2020 paper. 
'''

import numpy as np
from pandas.core.indexing import convert_from_missing_indexer_tuple
from scipy.stats import norm 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors
import sklearn.cluster 
from math import cos,sin,radians

from sklearn.utils.extmath import density

def ComputePR(P,Q,k):
    #Function to compute precision, and recall as in Sajjadi et al 2018 
    # (Code is heavily recycled from there)
    epsilon = 1e-10
    num_angles = 1001
    
    cluster_data = np.vstack([P,Q])
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=k,n_init=10)
    labels = kmeans.fit(cluster_data).labels_

    true_labels = labels[:len(P)]
    gen_labels  = labels[len(P):]

    true_bins = np.histogram(true_labels, bins=k,
                           range=[0, k], density=True)[0]
    gen_bins = np.histogram(gen_labels, bins=k,
                          range=[0, k], density=True)[0]



    # Compute slopes for linearly spaced angles between [0, pi/2]
    angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
    slopes = np.tan(angles)

    # Broadcast slopes so that second dimension will be states of the distribution
    slopes_2d = np.expand_dims(slopes, 1)

    # Broadcast distributions so that first dimension represents the angles
    ref_dist_2d = np.expand_dims(true_bins, 0)
    eval_dist_2d = np.expand_dims(gen_bins, 0)

    # Compute precision and recall for all angles in one step via broadcasting
    precision = np.minimum(ref_dist_2d*slopes_2d, eval_dist_2d).sum(axis=1)
    recall = precision / slopes

    # handle numerical instabilities leaing to precision/recall just above 1
    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)
    
    return precision, recall

def PRCover(P,Q,k):

    num_P = P.shape[0]
    num_Q = Q.shape[0]
    
    nbrs_P = NearestNeighbors(n_neighbors=(3*k), algorithm='kd_tree').fit(P)
    nbrs_Q = NearestNeighbors(n_neighbors=(3*k), algorithm='kd_tree').fit(Q)

    #returns KNN distances and indices for each data sample
    dist_P, ind_P = nbrs_P.kneighbors(P)
    dist_Q, ind_Q = nbrs_Q.kneighbors(Q)

    #Note that the knn returns the pt itself as 1NN so we discard first column
    dist_P = dist_P[:,1:]
    dist_Q = dist_Q[:,1:]
    ind_P  =  ind_P[:,1:]
    ind_Q  =  ind_Q[:,1:]

    p_sum = 0
    r_sum = 0

    for i in range(num_P):
        if PR_Cover_Indicator(P[i],Q,dist_P) == 1:
            p_sum += 1


    for j in range(num_Q): 
        if PR_Cover_Indicator(Q[i],P,dist_Q) == 1:
            r_sum += 1

    cover_precision = p_sum/num_P
    cover_recall    = r_sum/num_Q

    return cover_precision, cover_recall

def PR_Cover_Indicator(sample_pt, sample_set, k_nn_set):
    
    k = k_nn_set.shape[1] + 1
    num_nbrs = k/3
    num_pts  = sample_set.shape[0]

    set_pts_in_knn = 0 
    
    for i in range(num_pts):
        curr_dist = np.linalg.norm(sample_set[i] - sample_pt)
        
        if curr_dist <= k_nn_set[i][k-2]:
            set_pts_in_knn += 1
    
    if set_pts_in_knn >= num_nbrs:
        indicator_val = 1
    else:
        indicator_val = 0

    return indicator_val


def IPR_Indicator_Function(sample_pt, sample_set, k_nn_set):
    #Checks to see if the pt passed into this function lies within the knn sphere of any point in the set passed into the function  

    k = k_nn_set.shape[1]    
    val = 0 
    num_pts = sample_set.shape[0]


    for i in range(num_pts):
        curr_dist = np.linalg.norm(sample_set[i] - sample_pt)

        # Checks the last NN distance because if curr dist is less than k-NN since knn is monotonically increasing it is less than all other k-nn distances 
        if curr_dist <= k_nn_set[i][k-1]:
            val = 1 
            return val
        
    return val 

def ComputeIPR(P,Q,k):
    #Computes improved precision and recall as in the 2019 paper. 
    nbrs_P = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(P)
    nbrs_Q = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(Q)

    #returns KNN distances and indices for each data sample
    dist_P, ind_P = nbrs_P.kneighbors(P)
    dist_Q, ind_Q = nbrs_Q.kneighbors(Q)

    #Note that the knn returns the pt itself as 1NN so we discard first column
    dist_P = dist_P[:,1:]
    dist_Q = dist_Q[:,1:]
    ind_P  =  ind_P[:,1:]
    ind_Q  =  ind_Q[:,1:]

    #Assumes that the dimensionality of P and Q are the same
    N = P.shape[0]
    M = Q.shape[0]
    d = P.shape[1] 

    p_sum = 0
    r_sum = 0
    
    #Compute precision
    for i in range(M):
        return_val = IPR_Indicator_Function(Q[i],P,dist_P)
        if  return_val == 1:
            p_sum += 1
    precision = p_sum/M

    #Compute recall       
    for i in range(N):
        return_val = IPR_Indicator_Function(P[i],Q,dist_Q)
        if  return_val == 1:
             r_sum += 1
    recall = r_sum/N

    return precision, recall



def ComputeDC(P,Q,k):
    #Computes density and coverage as in Naeem et al 2020
    '''
    Note: that normally knn computation always includes 1-NN as the point itself. As this 
    leads to the first column of the distance matrix to always be 0 and the first column 
    of the index matrix to always be the row number in this function we discard the first 
    column for both matrices and thus whenever k is mentioned we are actually referring to 
    k-1. So if we input k =2 we are actually finding the 1-NN not including a point to itself
    '''
    nbrs_P = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(P)
    nbrs_Q = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(Q)

    dist_P, ind_P = nbrs_P.kneighbors(P)
    dist_Q, ind_Q = nbrs_Q.kneighbors(Q)

    #Note that the knn returns the pt itself as 1NN so we discard first column
    dist_P = dist_P[:,1:]
    dist_Q = dist_Q[:,1:]
    ind_P  =  ind_P[:,1:]
    ind_Q  =  ind_Q[:,1:]

    #Assumes that the dimensionality of P and Q are the same
    N = P.shape[0]
    M = Q.shape[0]
    d = P.shape[1]

    d_sum = 0
    c_sum = 0

    #Iterates through each generated sample and checks within how many real samples it lies  
    for i in range(N):
        for j in range(M):
            curr_dist = np.linalg.norm((Q[j]-P[i]))

            #checks if distance is within the k-nn distance
            if curr_dist <= dist_P[i][k-2]:
                d_sum += 1


    for i in range(N):
        for j in range(M):
            curr_dist = np.linalg.norm((Q[j]-P[i]))

            #checks if distance is within the k-nn distance
            if curr_dist <= dist_P[i][k-2]:
                c_sum += 1
                break

    
    density  = d_sum/((k-1)*M)
    coverage = c_sum/N 

    return density, coverage

def PlotData(P,Q, fig_num, plotstyle = '1d', save_fig = 'off'):
    #Takes the samples and plots them depending on the dimensionality
    dim_P = P.shape[1]
    dim_Q = Q.shape[1]

    
    if (dim_P == 1) and (dim_Q == 1):

        if plotstyle != '3d':
            #fig, ax = plt.subplots(figsize=(10,10))
            fig = plt.figure()
            ax = fig.add_subplot(figsize=(10,10))
            ax.set_xlabel('value')
            ax.set_ylabel('frequency')
            ax.set_title('Histogram of P (true data) and Q (gen data)')

            ax.hist(P, bins = 'auto', color='blue', alpha=0.5, label='P True distribution')
            ax.hist(Q, bins = 'auto', color='red' , alpha=0.5, label='Q Gen  distribution')
            ax.legend(['True distribution P','Generated distribution Q'])
            plt.legend()

            #Saves an image of the plot in the appropriate directory with appropriate naming.
            if save_fig == 'on':
                fig.savefig("Experiments/experiment%d.png"%(fig_num))

            plt.show()
        else:
            #Code to do 3d histograms
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')

            hist, bins = np.histogram(P, bins='auto')
            xs = (bins[:-1] + bins[1:])/2
            ax.bar(xs, hist, zs = 0, alpha=0.8, color='blue')

            hist, bins = np.histogram(Q, bins='auto')
            xs = (bins[:-1] + bins[1:])/2
            ax.bar(xs, hist, zs =  10, alpha=0.8, color ='red')
            
            #Saves an image of the plot in the appropriate directory with appropriate naming.
            if save_fig == 'on':
                fig.savefig("Experiments/experiment%d.png"%(fig_num))
    else:
        # assumes 2d plots
        P_x = P[:,0]
        P_y = P[:,1]
        Q_x = Q[:,0]
        Q_y = Q[:,1]

        fig, ax = plt.subplots()
        ax.scatter(P_x,P_y, color = 'blue')
        ax.scatter(Q_x,Q_y, color = 'red')
        ax.legend(['True Distribution P', 'Gen Distribution Q'])
        ax.set_title('Plotting 2d experiment of generated and real distributions')
        ax.set_ylabel('y axis')
        ax.set_xlabel('x axis')
        
        #Saves an image of the plot in the appropriate directory with appropriate naming.
        if save_fig == 'on':
                fig.savefig("Experiments/experiment%d.png"%(fig_num))
        
        plt.show()

def UniformData1D(n,m,a_P,b_P,a_Q,b_Q,r_seed):
    '''
    Takes in parameters m: num gen samples, n: num real samples, a_P: start of uniform dist for P, b_P: 
    end of uniform dist for P, (a_Q and b_Q start end pts for Q) and a random seed value to generate 
    2 sets of samples P and Q for real and gen distributions respectively.
    '''
    np.random.seed(r_seed)

    P = np.random.uniform(a_P, b_P, (n,1))
    Q = np.random.uniform(a_Q, b_Q, (m,1))

    return P, Q 
    
def UniformData2D(n, m, x1_P, x2_P, y1_P, y2_P, x1_Q, x2_Q, y1_Q, y2_Q, r_seed):
    '''
    Takes values of endpoints to define 2 rectangles (for P and Q) a uniform distribution is then sampled
    over the defined rectangular. Also takes how many number of samples there are for each sample set.
    '''
    np.random.seed(r_seed)

    P_x = np.random.uniform(x1_P, x2_P, (n,1))
    P_y = np.random.uniform(y1_P, y2_P, (n,1))

    Q_x = np.random.uniform(x1_Q, x2_Q, (m,1))
    Q_y = np.random.uniform(y1_Q, y2_Q, (m,1))

    P = np.hstack([P_x, P_y])
    Q = np.hstack([Q_x, Q_y])

    return P, Q

def Gaussian2D(n,m, x_P, y_P, x_Q, y_Q, std_P, std_Q, r_seed):
    '''
    takes the num samples, mean (x,y coord separately) and std of each distribution (P and Q respectively)
    and returns a 2d normal distribution in particular the x and y coord of the true and gen dist.
    '''

    np.random.seed(r_seed)

    P = np.random.multivariate_normal(np.array([x_P,y_P]),np.array([[std_P,0],[0, std_P]]),(n))
    Q = np.random.multivariate_normal(np.array([x_Q,y_Q]),np.array([[std_Q,0],[0, std_Q]]),(m))

    return P, Q

def Experiments():
# Set of experiments to be conducted
    r_seed = 7
    num_vals = 1000
    init_val = 0
    k = 3
    
    #1D point generator 
    
    #case 1, matching 1D dist
    k1 = k
    P1,Q1 = UniformData1D(1000,1000,0,10,0,10,r_seed)
    PlotData(P1,Q1)
    c1_precision, c1_recall = PRCover(P1,Q1,k1)
    density1, coverage1 = ComputeDC(P1,Q1,k1)
    p1, r1 = ComputePR(P1,Q1,k1)
    Ip1, Ir1 = ComputeIPR(P1,Q1,k1)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall)

    #case 2, disjoint 1D dist
    k2 = k
    P2,Q2 = UniformData1D(1000,1000,0,10,11,20,r_seed)
    PlotData(P2,Q2)
    c2_precision, c2_recall = PRCover(P2,Q2,k2)
    density2, coverage2 = ComputeDC(P2,Q2,k2)
    p2, r2 = ComputePR(P2,Q2,k2)
    Ip2, Ir2 = ComputeIPR(P2,Q2,k2)
    PlotResults(p2,r2,Ip2,Ir2,density2,coverage2,c2_precision,c2_recall)

    #case 3, overlapping 1D dist
    k3 = k
    P3,Q3 = UniformData1D(1000,1000,0,10,5,15,r_seed)
    PlotData(P3,Q3)
    c3_precision, c3_recall = PRCover(P3,Q3,k3)
    density3, coverage3 = ComputeDC(P3,Q3,k3)
    p3, r3 = ComputePR(P3,Q3,k3)
    Ip3, Ir3 = ComputeIPR(P3,Q3,k3)
    PlotResults(p3,r3,Ip3,Ir3,density3,coverage3,c3_precision,c3_recall)

    #case 4, matching 2D dist
    k4 = k
    P4, Q4 = UniformData2D(1000,1000,5,13,7,19,5,13,7,19,r_seed)
    PlotData(P2,Q2)
    c2_precision, c2_recall = PRCover(P2,Q2,k2)
    density2, coverage2 = ComputeDC(P2,Q2,k2)
    p2, r2 = ComputePR(P2,Q2,k2)
    Ip2, Ir2 = ComputeIPR(P2,Q2,k2)
    PlotResults(p2,r2,Ip2,Ir2,density2,coverage2,c2_precision,c2_recall)









def TestDataGenerator():
    x_vals = [-1,-1,-1,0,0,0,1,1,1]
    y_vals = [-1,0,1,-1,0,1,-1,0,1]
    a_list = np.linspace(225,-90,9) #list of angles starting from bottom corner point going clockwise
    radius = 1
    epsilon = 0.1

    #true_data  = np.array(([-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]))
    #test_data1 = np.array(([],[],[],[],[],[],[],[],[]))
    
    boundary_xvals = []
    boundary_yvals = []
    genx1 = []
    genx2 = []
    geny1 = []
    geny2 = []

    #Populate the arrays, the boundary values are defined by the radius of the circles, 
    for i in range(len(x_vals)):
        boundary_xvals.append(x_vals[i]+radius*cos(radians(a_list[i])))
        boundary_yvals.append(y_vals[i]+radius*sin(radians(a_list[i])))
    
        genx1.append(x_vals[i]+(radius-epsilon)*cos(radians(a_list[i])))
        geny1.append(y_vals[i]+(radius-epsilon)*sin(radians(a_list[i])))

        genx2.append(x_vals[i]+(radius+epsilon)*cos(radians(a_list[i])))
        geny2.append(y_vals[i]+(radius+epsilon)*sin(radians(a_list[i])))

    #convert all generated data points into arrays    
    xg1 = np.array(genx1)
    yg1 = np.array(geny1)
    xg2 = np.array(genx2)
    yg2 = np.array(geny2)

    true_data = np.vstack((np.array(x_vals),np.array(y_vals)))
    gen1_data = np.vstack((xg1,yg1))
    gen2_data = np.vstack((xg2,yg2))

    true_data = np.transpose(true_data)
    gen1_data = np.transpose(gen1_data)
    gen2_data = np.transpose(gen2_data)

    gen3_data = np.array(([(-0.99,-0.99),(-0.99,0),(-0.99,1),(0,-0.99),(0,0.01),(0,1.01),(1,-0.99),(1,0.01),(1,1.01)]))
    gen4_data = np.array(([(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]))
    gen5_data = np.array(([(-20,-20),(-20,0),(-20,20),(0,-20),(0,20),(0,0),(20,-20),(20,0),(20,20)]))

    return true_data, gen1_data, gen2_data, gen3_data, gen4_data, gen5_data

def PlotResults(precision, recall, I_precision, I_recall, density, coverage, c_precision, c_recall):

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.fill_between(recall, 0, precision, color='green')
    ax.set_title("Precision and Recall")
    ax.set_xlabel(r'Recall $ \beta $')
    ax.set_ylabel(r'Precision $ \alpha $')

    ax.text(0.65, 1.07, r'Density = %4.2f , Coverage = %4.2f' % (density, coverage), fontsize=12)
    ax.text(0.65, 1.02, r'I_precision = %4.2f , I_recall = %4.2f' % (I_precision, I_recall), fontsize=12)
    ax.text(0.65, 1.13, r'C_precision = %4.2f , C_recall = %4.2f' % (c_precision, c_recall), fontsize=12)

    plt.show()

def UnitTest2():

    nbrs = NearestNeighbors(n_neighbors=8, algorithm='kd_tree').fit(true_data)
    dist, ind = nbrs.kneighbors(true_data)
    dist = dist[:,1:]
    ind  = ind[:,1:]
    test_val = IPR_Indicator_Function(gen1_data[0],true_data,dist)

    print('nn')
    print(dist)
    print('test val')
    print(test_val)

def UnitTest3():
    p1,r1 = ComputeIPR(true_data,gen1_data,7)
    print('First PR score: ')
    print(p1, ' ', r1)

    p2,r2 = ComputeIPR(true_data,gen2_data,7)
    print('Second PR score: ')
    print(p2, ' ', r2)

    p3,r3 = ComputeIPR(true_data,gen3_data,7)
    print('Third PR score: ')
    print(p3, ' ', r3)
    
    p4,r4 = ComputeIPR(true_data,gen4_data,7)
    print('Fourth PR score: ')
    print(p4, ' ', r4)

    p5,r5 = ComputeIPR(true_data,gen5_data,7)
    print('Fifth PR score: ')
    print(p5, ' ', r5)

def UnitTest4():
    #Recreate figure 3e in paper by sajjadi et al 2018
    temp1 = np.ones((10,1))
    temp2 = np.ones((10,1))
    temp1 += 1 
    P3 = np.vstack([temp1,temp2])

    temp1 = np.ones((1,1))
    temp1 *= 2
    temp2 = np.ones((19,1))
    Q3 = np.vstack([temp1,temp2])

    k = 2

    precision, recall = ComputePR(P3, Q3, 20)
    density, coverage = ComputeDC(P3,Q3,k)
    I_precision, I_recall = ComputeIPR(P3,Q3,k)
    PlotResults(precision, recall, I_precision, I_recall, density, coverage)


if __name__ == "__main__":
    true_data, gen1_data, gen2_data, gen3_data, gen4_data, gen5_data = TestDataGenerator()

    print('Real Data:')
    print(true_data)

    print('First Generated Dataset:')
    print(gen1_data)

    print('Second Generated Dataset:')
    print(gen2_data)

    print('Third Generated Dataset:')
    print(gen3_data)

    print('Fourth Generated Dataset:')
    print(gen4_data)
    
    print('Fifth Generated Dataset:')
    print(gen5_data)
    
    p1,r1 = ComputeIPR(true_data,gen1_data,2)
    print('First PR score: ')
    print(p1, ' ', r1)

    p2,r2 = ComputeIPR(true_data,gen2_data,2)
    print('Second PR score: ')
    print(p2, ' ', r2)

    p3,r3 = ComputeIPR(true_data,gen3_data,2)
    print('Third PR score: ')
    print(p3, ' ', r3)
    
    p4,r4 = ComputeIPR(true_data,gen4_data,2)
    print('Fourth PR score: ')
    print(p4, ' ', r4)

    p5,r5 = ComputeIPR(true_data,gen5_data,2)
    print('Fifth PR score: ')
    print(p5, ' ', r5)

    print('Density and Coverage Tests')

    d1,c1 = ComputeDC(true_data,gen1_data,2)
    d2,c2 = ComputeDC(true_data,gen2_data,2)
    d3,c3 = ComputeDC(true_data,gen3_data,2)
    d4,c4 = ComputeDC(true_data,gen4_data,2)
    d5,c5 = ComputeDC(true_data,gen5_data,2)

    print('First DC score:')
    print(d1, ' ', c1)
    print('Second DC score:')
    print(d2, ' ', c2)
    print('Third DC score:')
    print(d3, ' ', c3)
    print('Fourth DC score:')
    print(d4, ' ', c4)
    print('Fifth DC score:')
    print(d5, ' ', c5)

    print('Test Case 6')
    gen6_data = np.array([(1000,1000),(1001,1000)])
    p6,r6 = ComputeIPR(true_data,gen6_data,2)
    d6,c6 = ComputeDC(true_data,gen6_data,2)
    print('precision and recall scores:')
    print(p6,'  ',r6)
    print('Density and coverage scores:')
    print(d6, '  ', c6)


'''
mean = 0 
std = 1

x = np.arange(-5,5,0.1)
y = norm.pdf(x,mean,std)

x_samples = np.arange(-5,5,0.01)
real_samples = np.random.normal(mean,std, 1000)
fake_samples = np.random.uniform(-1,1,1000)
gen1_samples = np.random.normal(mean+0.25,std,1000)

plt.figure(0)
plt.plot(x,y)

plt.figure(1)
plt.hist(real_samples,color='r',fill=False)
plt.hist(fake_samples,color='b',fill=True)
plt.hist(gen1_samples,color='g',fill=False)
plt.show()
'''