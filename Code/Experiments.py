from matplotlib.pyplot import figimage
from scipy.sparse import data
from DataGenerator import DataGenerator
import numpy as np
from GenerativeMetrics import ComputeDC, PRCover,PR_Cover_Indicator,ComputeIPR,ComputePR, IPR_Indicator_Function
from Plotter import PlotResults, PlotData

def ExperimentQueue():
    '''
    Main function to run all experiments
    '''
    #All experiments for reproducibility are intialized with the same random seed of 7
    r_seed = 7 
    k = 3
    fig_num = 36
    n = 1000
    m = 1000
    fig_num = Experiment2(r_seed,fig_num,k,n,m)
    fig_num = Experiment3(r_seed,fig_num,k,n,m)
    

def Experiment1(r_seed, fig_num, k, n, m):
    #First set of experiments will use the same k,n,m but just vary over the various uniform distributions
    #**********************************************************************************************
    DataSet = DataGenerator(r_seed)

    #1D uniform and matching dists
    fig_num += 1 
    P, Q = DataSet.UniformData1D(n,m,0,10,0,10)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,0,0,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 0,0,0,save_fig='on',quick_time='on')

    #1D uniform and disjoint dists
    fig_num += 1 
    P, Q = DataSet.UniformData1D(n,m,0,10,20,30)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,0,0,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 0,0,1,save_fig='on',quick_time='on')

    #1D uniform and overlapping dists, using sliding distributions 
    a1_list = [10,10,10,10,10,8,6,4,2,0]
    b1_list = [20,20,20,20,20,18,16,14,12,10]
    a2_list = [0,2,4,8,10,10,10,10,10,10]
    b2_list = [10,12,14,18,20,20,20,20,20,20] 

    for i in range(len(a1_list)):
        fig_num += 1
        P, Q = DataSet.UniformData1D(n,m,a1_list[i],b1_list[i],a2_list[i],b2_list[i])
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover  = PRCover(P,Q,k)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,0,0,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 0,0,2,save_fig='on',quick_time='on')

    #2D uniform and matching dists
    fig_num += 1 
    P, Q = DataSet.UniformData2D(n,m,0,10,0,10,0,10,0,10)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,1,1,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 1,1,0,save_fig='on',quick_time='on')

    #2D uniform and disjoint dists
    fig_num += 1 
    P, Q = DataSet.UniformData2D(n,m,0,10,0,10,20,30,20,30)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,1,1,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 1,1,1,save_fig='on',quick_time='on')

    #2D uniform and overlapping dists, using sliding distributions 
    Px1_list = [0,0,0,0,0,2,4,6,8,10]
    Px2_list = [10,10,10,10,10,12,14,16,18,20]
    Py1_list = [0,0,0,0,0,2,4,6,8,10]
    Py2_list = [10,10,10,10,10,12,14,16,18,20] 
    
    Qx1_list = [10,8,6,4,2,0,0,0,0,0]
    Qx2_list = [20,18,16,14,12,10,10,10,10,10]
    Qy1_list = [10,8,6,4,2,0,0,0,0,0]
    Qy2_list = [20,18,16,14,12,10,10,10,10,10,10] 

    for i in range(len(Px1_list)):
        fig_num += 1
        P, Q = DataSet.UniformData2D(n,m,Px1_list[i],Px2_list[i],Py1_list[i],Py2_list[i],Qx1_list[i],Qx2_list[i],Qy1_list[i],Qy2_list[i])
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover  = PRCover(P,Q,k)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,1,1,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 1,1,2,save_fig='on',quick_time='on')

    #3D uniform and matching dists
    fig_num += 1 
    P, Q = DataSet.UniformData3D(n,m,0,10,0,10,0,10,0,10,0,10,0,10)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,2,2,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 2,2,0,save_fig='on',quick_time='on')

    #3D uniform and disjoint dists
    fig_num += 1 
    P, Q = DataSet.UniformData3D(n,m,0,10,0,10,0,10,20,30,20,30,20,30)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,2,2,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 2,2,1,save_fig='on',quick_time='on')

    #3D uniform and overlapping dists, using sliding distributions 
    Px1_list = [0,0,0,0,0,2,4,6,8,10]
    Px2_list = [10,10,10,10,10,12,14,16,18,20]
    Py1_list = [0,0,0,0,0,2,4,6,8,10]
    Py2_list = [10,10,10,10,10,12,14,16,18,20] 
    Pz1_list = [0,0,0,0,0,2,4,6,8,10]
    Pz2_list = [10,10,10,10,10,12,14,16,18,20] 
    
    Qx1_list = [10,8,6,4,2,0,0,0,0,0]
    Qx2_list = [20,18,16,14,12,10,10,10,10,10]
    Qy1_list = [10,8,6,4,2,0,0,0,0,0]
    Qy2_list = [20,18,16,14,12,10,10,10,10,10,10] 
    Qz1_list = [10,8,6,4,2,0,0,0,0,0]
    Qz2_list = [20,18,16,14,12,10,10,10,10,10,10] 

    for i in range(len(Px1_list)):
        fig_num += 1
        P, Q = DataSet.UniformData3D(n,m,Px1_list[i],Px2_list[i],Py1_list[i],Py2_list[i],Pz1_list[i],Pz2_list[i],Qx1_list[i],Qx2_list[i],Qy1_list[i],Qy2_list[i],Qz1_list[i],Qz2_list[i])
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover  = PRCover(P,Q,k)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,2,2,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 2,2,2,save_fig='on',quick_time='on')

    return fig_num

def Experiment2(r_seed, fig_num, k, n, m):
    '''
    Various 2D Gaussian distributions  
    '''
    DataSet = DataGenerator(r_seed)

    #2D Gaussian and matching dists
    fig_num += 1 
    P, Q = DataSet.Gaussian2D(n,m,0,0,0,0,1,1)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,3,3,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 3, 3, 0,save_fig='on',quick_time='on')

    #2D Gaussian and disjoint dists
    fig_num += 1 
    P, Q = DataSet.Gaussian2D(n,m,0,0,10,10,1,1)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,3,3,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 3, 3, 1,save_fig='on',quick_time='on')

    #2D Gaussian and overlapping dists with sliding dists
    Px_list = [0,0,0,0,0,0.4,0.8,1.2,1.6,2.0]
    Py_list = [0,0,0,0,0,0.4,0.8,1.2,1.6,2.0]
    Qx_list = [2,1.6,1.2,0.8,0.4,0,0,0,0,0]
    Qy_list = [2,1.6,1.2,0.8,0.4,0,0,0,0,0]

    std_P = 1
    std_Q = 1

    for i in range(len(Px_list)):
        fig_num += 1 
        P, Q = DataSet.Gaussian2D(n,m,Px_list[i],Py_list[i],Qx_list[i],Qy_list[i],std_P,std_Q)
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover  = PRCover(P,Q,k)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,3,3,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 3, 3, 2,save_fig='on',quick_time='on')

    return fig_num

def Experiment3(r_seed, fig_num, k, n, m):
    '''
    Various 3D Gaussian distributions  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Gaussian and matching dists
    fig_num += 1 
    P, Q = DataSet.Gaussian3D(n,m,0,0,0,0,0,0,1,1)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,4,4,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 4, 4, 0,save_fig='on',quick_time='on')

    #3D Gaussian and disjoint dists
    fig_num += 1 
    P, Q = DataSet.Gaussian3D(n,m,0,0,0,10,10,10,1,1)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover  = PRCover(P,Q,k)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,4,4,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 4, 4, 1,save_fig='on',quick_time='on')

    #2D Gaussian and overlapping dists with sliding dists
    Px_list = [0,0,0,0,0,0.4,0.8,1.2,1.6,2.0]
    Py_list = [0,0,0,0,0,0.4,0.8,1.2,1.6,2.0]
    Pz_list = [0,0,0,0,0,0.4,0.8,1.2,1.6,2.0]
    Qx_list = [2,1.6,1.2,0.8,0.4,0,0,0,0,0]
    Qy_list = [2,1.6,1.2,0.8,0.4,0,0,0,0,0]
    Qz_list = [2,1.6,1.2,0.8,0.4,0,0,0,0,0]

    std_P = 1
    std_Q = 1

    for i in range(len(Px_list)):
        fig_num += 1 
        P, Q = DataSet.Gaussian3D(n,m,Px_list[i],Py_list[i],Pz_list[i],Qx_list[i],Qy_list[i],Qz_list[i],std_P,std_Q)
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover  = PRCover(P,Q,k)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,4,4,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, fig_num, 4, 4, 2,save_fig='on',quick_time='on')

    return fig_num 