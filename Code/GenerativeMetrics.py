'''
    Author: Fasil Cheema
    Purpose: Compute Density and Coverage as described in 2020 paper. 
'''

import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors
import sklearn.cluster 
from math import cos,sin,radians

def ComputePR(P,Q,k):
    epsilon = 1e-10
    num_angles = 10001
    
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



def IPR_Indicator_Function(sample_pt, sample_set,k_nn_set):
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

    for i in range(N):
        for j in range(M):
            curr_dist = np.linalg.norm((Q[j]-P[i]))

            if curr_dist <= dist_P[i][k-2]:
                d_sum += 1


    for i in range(N):
        for j in range(M):
            curr_dist = np.linalg.norm((Q[j]-P[i]))

            if curr_dist <= dist_P[i][k-2]:
                c_sum += 1
                break

    
    density  = d_sum/((k-1)*M)
    coverage = c_sum/N 

    return density, coverage

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

def PlotPR(precision, recall):
    plt.plot(precision,recall)


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