'''
    Author: Fasil Cheema
    Purpose: Computes several generative metrics such as density and coverage, precision and recall, 
             improved prevision and improved recall, and our proposed metric cover precision and cover recall. 
'''

import numpy as np
from numpy.lib.npyio import save
from pandas.core.indexing import convert_from_missing_indexer_tuple
from scipy.stats import norm 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import NearestNeighbors
import sklearn.cluster 
from math import cos,sin,radians,pi

from sklearn.utils.extmath import density

def ComputePR(P,Q,k):
    # Function to compute precision, and recall as in Sajjadi et al 2018 
    # (Code is heavily recycled from there)
    epsilon = 1e-10
    num_angles = 1001
    
    cluster_data = np.vstack([P,Q])
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=(k),n_init=10)
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

    # Handle numerical instabilities leaing to precision/recall just above 1
    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)
    
    return precision, recall

def PRCover(P,Q,k):
    # Computes the proposed cover precision and cover recall metrics

    # Obtains the number of samples in both samples sets P and Q
    num_P = P.shape[0]
    num_Q = Q.shape[0]

    # C factor is simply an integer where k' = Ck (originally set to 3)
    C = 9

    # Computes the NN of both P and Q
    nbrs_P = NearestNeighbors(n_neighbors=(C*k)+1, algorithm='kd_tree').fit(P)
    nbrs_Q = NearestNeighbors(n_neighbors=(C*k)+1, algorithm='kd_tree').fit(Q)

    # Returns KNN distances and indices for each data sample
    dist_P, ind_P = nbrs_P.kneighbors(P)
    dist_Q, ind_Q = nbrs_Q.kneighbors(Q)

    # Note that the knn returns the pt itself as 1NN so we discard first column
    dist_P = dist_P[:,1:]
    dist_Q = dist_Q[:,1:]
    ind_P  =  ind_P[:,1:]
    ind_Q  =  ind_Q[:,1:]

    # Intialize metric counter
    p_sum = 0
    r_sum = 0

    # Iterates through sample set P and checks if the number of set pts within the sample pt k-NN are above the desired number
    for i in range(num_P):
        return_val = PR_Cover_Indicator(P[i],Q,dist_P[i], C)
        if return_val == 1:
            p_sum += 1

    # Computes cover_precision (num times k-nn ball for pt is sufficiently mixed divided )
    cover_precision = p_sum/num_P

    # Iterates through sample set Q and checks if the number of set pts within the sample pt k-NN are above the desired number
    for j in range(num_Q): 
        return_val = PR_Cover_Indicator(Q[j],P,dist_Q[j], C)
        if return_val == 1:
            r_sum += 1

    # Computes cover_recall (num times k-nn ball for pt is sufficiently mixed divided )
    cover_recall = r_sum/num_Q

    return cover_precision, cover_recall

def PR_Cover_Indicator(sample_pt, sample_set, k_nn_set, C):
    # Indicator function that checks if the number of pts from the set that lie within the k-NN ball of
    # the input point exceeds the required number of neighbors ( which is based off of the C factor k' = Ck)

    # Obtain important info such as choice of k, num_nbrs which is the min num of pts within a k-nn ball
    k = len(k_nn_set)
    num_nbrs = k/C
    num_pts  = sample_set.shape[0]

    # Initialize counter for num pts within k-nn ball 
    set_pts_in_knn = 0 
    
    # Iterate through each pt in set and check if it lies within main pt's k-nn ball if so add to count
    for i in range(num_pts):
        curr_dist = np.linalg.norm(sample_set[i] - sample_pt)
        
        if curr_dist <= k_nn_set[k-1]:
            set_pts_in_knn += 1
    
    # Checks if the number of pts that are within k-nn ball of main pt is above threshold (num_nbrs) if so return 1
    if set_pts_in_knn >= num_nbrs:
        indicator_val = 1
    else:
        indicator_val = 0

    return indicator_val


def IPR_Indicator_Function(sample_pt, sample_set, k_nn_set):
    # Checks to see if the pt passed into this function lies within the knn sphere of any point in the set passed into the function  

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
    # Computes improved precision and recall as in the 2019 paper. 
    nbrs_P = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(P)  # We use k+1 because we want k DISTINCT NN and because the algorithm always includes the point itself as its 1-NN (which we discard) we use (k+1)-NN 
    nbrs_Q = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(Q)

    # Returns KNN distances and indices for each data sample
    dist_P, ind_P = nbrs_P.kneighbors(P)
    dist_Q, ind_Q = nbrs_Q.kneighbors(Q)

    # Note that the knn returns the pt itself as 1NN so we discard first column
    dist_P = dist_P[:,1:]
    dist_Q = dist_Q[:,1:]
    ind_P  =  ind_P[:,1:]
    ind_Q  =  ind_Q[:,1:]

    # Assumes that the dimensionality of P and Q are the same
    N = P.shape[0]
    M = Q.shape[0]
    d = P.shape[1] 

    # Initialize counter for precision and recall
    p_sum = 0
    r_sum = 0
    
    # Compute precision
    for i in range(M):
        return_val = IPR_Indicator_Function(Q[i],P,dist_P)
        if  return_val == 1:
            p_sum += 1
    precision = p_sum/M

    # Compute recall       
    for i in range(N):
        return_val = IPR_Indicator_Function(P[i],Q,dist_Q)
        if  return_val == 1:
             r_sum += 1
    recall = r_sum/N

    return precision, recall


def ComputeDC(P,Q,k):
    # Computes density and coverage as in Naeem et al 2020
    '''
    Note: that normally knn computation always includes 1-NN as the point itself. As this 
    leads to the first column of the distance matrix to always be 0 and the first column 
    of the index matrix to always be the row number in this function we discard the first 
    column for both matrices and thus whenever k is mentioned we are actually referring to 
    k-1. So if we input k =2 we are actually finding the 1-NN not including a point to itself
    '''
    # Compute NN for P and Q and find distance and index matrices
    nbrs_P = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(P) # Use k+1 NN because 1-NN is always point itself
    nbrs_Q = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(Q)
    dist_P, ind_P = nbrs_P.kneighbors(P)
    dist_Q, ind_Q = nbrs_Q.kneighbors(Q)

    # Note that the knn returns the pt itself as 1NN so we discard first column
    dist_P = dist_P[:,1:]
    dist_Q = dist_Q[:,1:]
    ind_P  =  ind_P[:,1:]
    ind_Q  =  ind_Q[:,1:]

    # Assumes that the dimensionality of P and Q are the same
    N = P.shape[0]
    M = Q.shape[0]
    d = P.shape[1]

    # Initialize density and coverage counter
    d_sum = 0
    c_sum = 0

    # Iterates through each generated sample and checks within how many real samples it lies  
    for i in range(N):
        for j in range(M):
            curr_dist = np.linalg.norm((Q[j]-P[i]))

            # Checks if distance is within the k-nn distance
            if curr_dist <= dist_P[i][k-2]:
                d_sum += 1

    # Iterates through each real sample and checks within how many generated samples it lies  
    for i in range(N):
        for j in range(M):
            curr_dist = np.linalg.norm((Q[j]-P[i]))

            # Checks if distance is within the k-nn distance
            if curr_dist <= dist_P[i][k-2]:
                c_sum += 1
                break

    # Compute density and coverage by dividing count by appropriate number
    density  = d_sum/((k-1)*M)
    coverage = c_sum/N 

    return density, coverage

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
