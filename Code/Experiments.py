from mimetypes import init
from tracemalloc import start
from matplotlib.pyplot import figimage
from scipy.sparse import data
from DataGenerator import DataGenerator
import numpy as np
import math
from GenerativeMetrics import ComputeDC, PRCover,PR_Cover_Indicator,ComputeIPR,ComputePR, IPR_Indicator_Function, ComputeTruePR
from Plotter import PlotResults, PlotData, PlotManifolds, PlotPrecisionConvergence, PlotRecallConvergence

def ExperimentQueue():
    '''
    Main function to run all experiments
    '''
    #All experiments for reproducibility are intialized with the same random seed of 7
    fig_num = 72


    #All experiments for reproducibility are intialized with the same random seed of 7
    #k for k-nearest neighbors algorithms. n,m are true dist sample sizes and gen sample sizes respectively.
    #C is a constant for setting an appropriate value of k' = Ck for the PR cover algorithm
    print(fig_num)
    r_seed = 7 
    k = 3 
    C = 3
    n = 1500
    m = 1500
    
    #Constant for choice of k (see theory approx ln(8/delta))
    C_0 = 6

    #For convergence experiments due to C_0 causing a base value for k, we need to start at a number of samples > k. Adjust accordingly to C_0
    start_val = 15

    num_samples = 1000

    #fig_num = Experiment1(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment2(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment3(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment4(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment5(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment6(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment7(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment8(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment9(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment10(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment11(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment12(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment13(r_seed,fig_num,k,n,m,C)
    #fig_num = Experiment14(r_seed,fig_num,k,n,m,C)
    #fig_num = MeasureConvergenceExperiment1(r_seed,num_samples,fig_num,C,C_0, start_val)
    #fig_num = MeasureConvergenceExperiment2(r_seed,num_samples,fig_num,C,C_0, start_val)
    #fig_num = MeasureConvergenceExperiment3(r_seed,num_samples,fig_num,C,C_0, start_val)
    fig_num = MeasureConvergenceExperiment(r_seed,num_samples,fig_num,C,C_0,start_val,4)
    print('finished 4d experiments')
    fig_num = MeasureConvergenceExperiment(r_seed,num_samples,fig_num,C,C_0,start_val,5)
    print('finished 5d experiments')
    fig_num = MeasureConvergenceExperiment(r_seed,num_samples,fig_num,C,C_0,start_val,6)
    print('finished 6d experiments')
    fig_num = MeasureConvergenceExperiment(r_seed,num_samples,fig_num,C,C_0,start_val,7)
    print('finished 7d experiments')
    fig_num = MeasureConvergenceExperiment(r_seed,num_samples,fig_num,C,C_0,start_val,8)
    print('finished 8d experiments')

def Experiment1(r_seed, fig_num, k, n, m, C):
    #First set of experiments will use the same k,n,m but just vary over the various uniform distributions
    #**********************************************************************************************
    DataSet = DataGenerator(r_seed)

    # #1D uniform and matching dists
    # fig_num += 1 
    # P, Q = DataSet.UniformData1D(n,m,0,10,0,10)
    # precision, recall = ComputePR(P,Q,k)
    # p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn  = PRCover(P,Q,k,C)
    # i_precision, i_recall = ComputeIPR(P,Q,k)
    # density, coverage = ComputeDC(P,Q,k)
    # PlotData(P,Q,fig_num,0,0,0,plotstyle='1d', save_fig='on',quick_time='on')
    # PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 0,0,0,save_fig='on',quick_time='on')
    # PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    # #1D uniform and disjoint dists
    # fig_num += 1 
    # P, Q = DataSet.UniformData1D(n,m,0,10,20,30)
    # precision, recall = ComputePR(P,Q,k)
    # p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn  = PRCover(P,Q,k,C)
    # i_precision, i_recall = ComputeIPR(P,Q,k)
    # density, coverage = ComputeDC(P,Q,k)
    # PlotData(P,Q,fig_num,0,0,1,plotstyle='1d', save_fig='on',quick_time='on')
    # PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 0,0,1,save_fig='on',quick_time='on')
    # PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    # #1D uniform and overlapping dists, using sliding distributions 
    # a1_list = [10,10,10,10,10,8,6,4,2,0]
    # b1_list = [20,20,20,20,20,18,16,14,12,10]
    # a2_list = [0,2,4,8,10,10,10,10,10,10]
    # b2_list = [10,12,14,18,20,20,20,20,20,20] 

    # for i in range(len(a1_list)):
    #     fig_num += 1
    #     P, Q = DataSet.UniformData1D(n,m,a1_list[i],b1_list[i],a2_list[i],b2_list[i])
    #     precision, recall = ComputePR(P,Q,k)
    #     p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn = PRCover(P,Q,k,C)
    #     i_precision, i_recall = ComputeIPR(P,Q,k)
    #     density, coverage = ComputeDC(P,Q,k)
    #     PlotData(P,Q,fig_num,0,0,2,plotstyle='1d', save_fig='on',quick_time='on')
    #     PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 0,0,2,save_fig='on',quick_time='on')
    #     PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    # #2D uniform and matching dists
    # fig_num += 1 
    # P, Q = DataSet.UniformData2D(n,m,0,10,0,10,0,10,0,10)
    # precision, recall = ComputePR(P,Q,k)
    # p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn = PRCover(P,Q,k,C)
    # i_precision, i_recall = ComputeIPR(P,Q,k)
    # density, coverage = ComputeDC(P,Q,k)
    # PlotData(P,Q,fig_num,1,1,0,plotstyle='1d', save_fig='on',quick_time='on')
    # PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 1,1,0,save_fig='on',quick_time='on')
    # PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    # #2D uniform and disjoint dists
    # fig_num += 1 
    # P, Q = DataSet.UniformData2D(n,m,0,10,0,10,20,30,20,30)
    # precision, recall = ComputePR(P,Q,k)
    # p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    # i_precision, i_recall = ComputeIPR(P,Q,k)
    # density, coverage = ComputeDC(P,Q,k)
    # PlotData(P,Q,fig_num,1,1,1,plotstyle='1d', save_fig='on',quick_time='on')
    # PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 1,1,1,save_fig='on',quick_time='on')
    # PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    # #2D uniform and overlapping dists, using sliding distributions 
    # Px1_list = [0,0,0,0,0,2,4,6,8,10]
    # Px2_list = [10,10,10,10,10,12,14,16,18,20]
    # Py1_list = [0,0,0,0,0,2,4,6,8,10]
    # Py2_list = [10,10,10,10,10,12,14,16,18,20] 
    
    # Qx1_list = [10,8,6,4,2,0,0,0,0,0]
    # Qx2_list = [20,18,16,14,12,10,10,10,10,10]
    # Qy1_list = [10,8,6,4,2,0,0,0,0,0]
    # Qy2_list = [20,18,16,14,12,10,10,10,10,10,10] 

    # for i in range(len(Px1_list)):
    #     fig_num += 1
    #     P, Q = DataSet.UniformData2D(n,m,Px1_list[i],Px2_list[i],Py1_list[i],Py2_list[i],Qx1_list[i],Qx2_list[i],Qy1_list[i],Qy2_list[i])
    #     precision, recall = ComputePR(P,Q,k)
    #     p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    #     i_precision, i_recall = ComputeIPR(P,Q,k)
    #     density, coverage = ComputeDC(P,Q,k)
    #     PlotData(P,Q,fig_num,1,1,2,plotstyle='1d', save_fig='on',quick_time='on')
    #     PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 1,1,2,save_fig='on',quick_time='on')
    #     PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D uniform and matching dists
    # fig_num += 1 
    # P, Q = DataSet.UniformData3D(n,m,0,10,0,10,0,10,0,10,0,10,0,10)
    # precision, recall = ComputePR(P,Q,k)
    # p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    # i_precision, i_recall = ComputeIPR(P,Q,k)
    # density, coverage = ComputeDC(P,Q,k)
    # PlotData(P,Q,fig_num,2,2,0,plotstyle='1d', save_fig='on',quick_time='on')
    # PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 2,2,0,save_fig='on',quick_time='on')
    # PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    # #3D uniform and disjoint dists
    # fig_num += 1 
    # P, Q = DataSet.UniformData3D(n,m,0,10,0,10,0,10,20,30,20,30,20,30)
    # precision, recall = ComputePR(P,Q,k)
    # p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    # i_precision, i_recall = ComputeIPR(P,Q,k)
    # density, coverage = ComputeDC(P,Q,k)
    # PlotData(P,Q,fig_num,2,2,1,plotstyle='1d', save_fig='on',quick_time='on')
    # PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 2,2,1,save_fig='on',quick_time='on')
    # PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

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
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,2,2,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 2,2,2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num

def Experiment2(r_seed, fig_num, k, n, m, C):
    '''
    Various 2D Gaussian distributions  
    '''
    DataSet = DataGenerator(r_seed)

    #2D Gaussian and matching dists
    fig_num += 1 
    P, Q = DataSet.Gaussian2D(n,m,0,0,0,0,1,1)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,3,3,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 3, 3, 0,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #2D Gaussian and disjoint dists
    fig_num += 1 
    P, Q = DataSet.Gaussian2D(n,m,0,0,10,10,1,1)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,3,3,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 3, 3, 1,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

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
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,3,3,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 3, 3, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num

def Experiment3(r_seed, fig_num, k, n, m, C):
    '''
    Various 3D Gaussian distributions  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Gaussian and matching dists
    fig_num += 1 
    P, Q = DataSet.Gaussian3D(n,m,0,0,0,0,0,0,1,1)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,4,4,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 4, 4, 0,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Gaussian and disjoint dists
    fig_num += 1 
    P, Q = DataSet.Gaussian3D(n,m,0,0,0,10,10,10,1,1)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,4,4,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 4, 4, 1,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

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
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,4,4,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 4, 4, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num 

def Experiment4(r_seed, fig_num, k, n, m, C):
    '''
    Various 2D Discs  
    '''
    DataSet = DataGenerator(r_seed)

    #2D Discs and matching dists
    fig_num += 1 
    P, Q = DataSet.Disc2D(n,m,0,5,0,0,0,5,0,0)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,5,5,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 5, 5, 0,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #2D Discs and disjoint dists
    fig_num += 1 
    P, Q = DataSet.Disc2D(n,m,0,5,0,0,0,5,30,30)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,5,5,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 5, 5, 1,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #2D Discs and overlapping dists with sliding dists
    Px_list = [0,0,0,0,0,2,4,6,8,10]
    Py_list = [0,0,0,0,0,2,4,6,8,10]
    Qx_list = [10,8,6,4,2,0,0,0,0,0]
    Qy_list = [10,8,6,4,2,0,0,0,0,0]

    for i in range(len(Px_list)):
        fig_num += 1 
        P, Q = DataSet.Disc2D(n,m,0,5,Px_list[i],Py_list[i],0,5,Qx_list[i],Qy_list[i])
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,5,5,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 5, 5, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num

def Experiment5(r_seed, fig_num, k, n, m, C):
    '''
    Various 3D Spheres  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Spheres and matching dists
    fig_num += 1 
    P, Q = DataSet.Sphere(n,m,0,5,0,0,0,0,5,0,0,0)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,6,6,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 6, 6, 0,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Spheres and disjoint dists
    fig_num += 1 
    P, Q = DataSet.Sphere(n,m,0,5,0,0,0,0,5,30,30,30)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,6,6,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 6, 6, 1,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Spheres and overlapping dists with sliding dists
    Px_list = [0,0,0,0,0,2,4,6,8,10]
    Py_list = [0,0,0,0,0,2,4,6,8,10]
    Pz_list = [0,0,0,0,0,2,4,6,8,10]
    Qx_list = [10,8,6,4,2,0,0,0,0,0]
    Qy_list = [10,8,6,4,2,0,0,0,0,0]
    Qz_list = [10,8,6,4,2,0,0,0,0,0]

    for i in range(len(Px_list)):
        fig_num += 1 
        P, Q = DataSet.Sphere(n,m,0,5,Px_list[i],Py_list[i],Pz_list[i],0,5,Qx_list[i],Qy_list[i],Qz_list[i])
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,6,6,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 6, 6, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num 

def Experiment5(r_seed, fig_num, k, n, m, C):
    '''
    Various Doughnuts  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Doughnuts and matching dists
    fig_num += 1 
    P, Q = DataSet.Doughnut(n,m,5,0.5,0,0,0,5,0.5,0,0,0)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,7,7,0,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 7, 7, 0,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Doughnuts and disjoint dists
    fig_num += 1 
    P, Q = DataSet.Doughnut(n,m,5,0.5,0,0,0,5,0.5,30,30,30)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,7,7,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 7, 7, 1,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Doughnuts and overlapping dists with sliding dists
    Px_list = [0,0,0,0,0,2,4,6,8,10]
    Py_list = [0,0,0,0,0,2,4,6,8,10]
    Pz_list = [0,0,0,0,0,2,4,6,8,10]
    Qx_list = [10,8,6,4,2,0,0,0,0,0]
    Qy_list = [10,8,6,4,2,0,0,0,0,0]
    Qz_list = [10,8,6,4,2,0,0,0,0,0]

    for i in range(len(Px_list)):
        fig_num += 1 
        P, Q = DataSet.Doughnut(n,m,5,0.5,Px_list[i],Py_list[i],Pz_list[i],5,0.5,Qx_list[i],Qy_list[i],Qz_list[i])
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,7,7,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 7, 7, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num 

def Experiment6(r_seed, fig_num, k, n, m, C):
    '''
    Various 3D Spheres and Doughnuts  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Sphere and doughnut overlapping
    fig_num += 1 
    P, _ = DataSet.Sphere(n,0,0,5,0,0,0,0,0,0,0,0)
    _, Q = DataSet.Doughnut(0,m,0,0,0,0,0,5.5,0.5,0,0,0)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,6,7,2,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 6, 7, 2,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Sphere and Doughnut overlapping with sliding dists
    Px_list = [0,0,0,0,0,2,4,6,8,10]
    Py_list = [0,0,0,0,0,2,4,6,8,10]
    Pz_list = [0,0,0,0,0,2,4,6,8,10]
    Qx_list = [10,8,6,4,2,0,0,0,0,0]
    Qy_list = [10,8,6,4,2,0,0,0,0,0]
    Qz_list = [10,8,6,4,2,0,0,0,0,0]

    for i in range(len(Px_list)):
        fig_num += 1 
        P, _ = DataSet.Sphere(n,0,0,5,Px_list[i],Py_list[i],Pz_list[i],0,0,0,0,0)
        _, Q = DataSet.Doughnut(0,m,0,0,0,0,0,5.5,0.5,Qx_list[i],Qy_list[i],Qz_list[i])
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,6,7,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 6, 7, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num 

def Experiment7(r_seed, fig_num, k, n, m, C):
    '''
    Various 3D Spheres and 3D Gaussians (std is always 5 to match radius of sphere)  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Sphere and 3D Gaussians overlapping
    fig_num += 1 
    P, _ = DataSet.Sphere(n,0,0,5,0,0,0,0,0,0,0,0)
    _, Q = DataSet.Gaussian3D(0,m,0,0,0,0,0,0,0,5)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,6,4,2,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 6, 4, 2,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Sphere and Doughnut overlapping with sliding dists
    Px_list = [0,0,0,0,0,2,4,6,8,10]
    Py_list = [0,0,0,0,0,2,4,6,8,10]
    Pz_list = [0,0,0,0,0,2,4,6,8,10]
    Qx_list = [10,8,6,4,2,0,0,0,0,0]
    Qy_list = [10,8,6,4,2,0,0,0,0,0]
    Qz_list = [10,8,6,4,2,0,0,0,0,0]

    for i in range(len(Px_list)):
        fig_num += 1 
        P, _ = DataSet.Sphere(n,0,0,5,Px_list[i],Py_list[i],Pz_list[i],0,0,0,0,0)
        _, Q = DataSet.Gaussian3D(0,m,0,0,0,Qx_list[i],Qy_list[i],Qz_list[i],0,5)
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,6,4,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 6, 4, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num 

def Experiment8(r_seed, fig_num, k, n, m, C):
    '''
    Various Doughnuts and 3D Gaussians (std is always 5 to match radius of sphere)  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Doughnut and 3D Gaussians overlapping
    fig_num += 1 
    P, _ = DataSet.Doughnut(n,0,4.5,0.5,0,0,0,0,0,0,0,0)
    _, Q = DataSet.Gaussian3D(0,m,0,0,0,0,0,0,0,5)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,7,4,2,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 7, 4, 2,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Gaussian and Doughnut overlapping with sliding dists
    Px_list = [0,0,0,0,0,2,4,6,8,10]
    Py_list = [0,0,0,0,0,2,4,6,8,10]
    Pz_list = [0,0,0,0,0,2,4,6,8,10]
    Qx_list = [10,8,6,4,2,0,0,0,0,0]
    Qy_list = [10,8,6,4,2,0,0,0,0,0]
    Qz_list = [10,8,6,4,2,0,0,0,0,0]

    for i in range(len(Px_list)):
        fig_num += 1 
        P, _ = DataSet.Doughnut(n,0,4.5,0.5,Px_list[i],Py_list[i],Pz_list[i],0,0,0,0,0)
        _, Q = DataSet.Gaussian3D(0,m,0,0,0,Qx_list[i],Qy_list[i],Qz_list[i],0,5)
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,7,4,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 7, 4, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num 

def Experiment9(r_seed, fig_num, k, n, m, C):
    '''
    Various 3D Cubes and 3D Gaussians  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Uniform cube and 3D Gaussians overlapping
    fig_num += 1 
    P, _ = DataSet.UniformData3D(n,0,-5,5,-5,5,-5,5,0,0,0,0,0,0)
    _, Q = DataSet.Gaussian3D(0,m,0,0,0,0,0,0,0,5)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,2,4,2,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 2, 4, 2,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Cube and Gaussians overlapping with sliding dists
    Px_list = [0,0,0,0,0,2,4,6,8,10]
    Py_list = [0,0,0,0,0,2,4,6,8,10]
    Pz_list = [0,0,0,0,0,2,4,6,8,10]
    Qx_list = [10,8,6,4,2,0,0,0,0,0]
    Qy_list = [10,8,6,4,2,0,0,0,0,0]
    Qz_list = [10,8,6,4,2,0,0,0,0,0]

    for i in range(len(Px_list)):
        fig_num += 1 
        P, _ = DataSet.UniformData3D(n,0,Px_list[i]-5,Px_list[i]+5,Py_list[i]-5,Py_list[i]+5,Pz_list[i]-5,Pz_list[i]+5,0,0,0,0,0,0)
        _, Q = DataSet.Gaussian3D(0,m,0,0,0,Qx_list[i],Qy_list[i],Qz_list[i],0,5)
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,2,4,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 2, 4, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num 

def Experiment10(r_seed, fig_num, k, n, m, C):
    '''
    Various 3D Cubes and Spheres  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Uniform cube and Sphere overlapping
    fig_num += 1 
    P, _ = DataSet.UniformData3D(n,0,-5,5,-5,5,-5,5,0,0,0,0,0,0)
    _, Q = DataSet.Sphere(0,m,0,0,0,0,0,0,5,0,0,0)
    precision, recall = ComputePR(P,Q,k)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,2,6,2,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 2, 6, 2,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    #3D Cube and Sphere overlapping with sliding dists
    Px_list = [0,0,0,0,0,2,4,6,8,10]
    Py_list = [0,0,0,0,0,2,4,6,8,10]
    Pz_list = [0,0,0,0,0,2,4,6,8,10]
    Qx_list = [10,8,6,4,2,0,0,0,0,0]
    Qy_list = [10,8,6,4,2,0,0,0,0,0]
    Qz_list = [10,8,6,4,2,0,0,0,0,0]

    for i in range(len(Px_list)):
        fig_num += 1 
        P, _ = DataSet.UniformData3D(n,0,Px_list[i]-5,Px_list[i]+5,Py_list[i]-5,Py_list[i]+5,Pz_list[i]-5,Pz_list[i]+5,0,0,0,0,0,0)
        _, Q = DataSet.Sphere(0,m,0,0,0,0,0,0,5,Qx_list[i],Qy_list[i],Qz_list[i])
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,2,6,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 2, 6, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num

def Experiment11(r_seed, fig_num, k, n, m, C):
    '''
    Various 3D Cubes rotated and Spheres  
    '''
    DataSet = DataGenerator(r_seed)

    #3D Uniform cube and Sphere overlapping with various rotations on the cube
    rx_list = [0,0,45,0,45,45,45]
    ry_list = [45,0,0,45,0,45,45]
    rz_list = [0,45,0,45,45,0,45]

    for i in range(len(rx_list)):
        fig_num += 1 
        P, _ = DataSet.UniformData3D(n,0,-5,5,-5,5,-5,5,0,0,0,0,0,0)
        _, Q = DataSet.Sphere(0,m,0,0,0,0,0,0,5,0,0,0)
        P = DataSet.Rotate3D(P,rx_list[i],ry_list[i],rz_list[i],0,0,0)
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,2,6,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 2, 6, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num

def Experiment12(r_seed, fig_num, k, n, m, C):
    '''
    Various Doughnuts rotated and Spheres  
    '''
    DataSet = DataGenerator(r_seed)

    #Doughnut and Sphere overlapping with various rotations on the doughnut
    rx_list = [0,0,45,0,45,45,45]
    ry_list = [45,0,0,45,0,45,45]
    rz_list = [0,45,0,45,45,0,45]

    for i in range(len(rx_list)):
        fig_num += 1 
        P, _ = DataSet.Doughnut(n,0,5.5,0.5,0,0,0,0,0,0,0,0)
        _, Q = DataSet.Sphere(0,m,0,0,0,0,0,0,5,0,0,0)
        P = DataSet.Rotate3D(P,rx_list[i],ry_list[i],rz_list[i],0,0,0)
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,7,6,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 7, 6, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num

def Experiment13(r_seed, fig_num, k, n, m, C):
    '''
    Various Doughnuts rotated with other doughnuts  
    '''
    #Initialize DataGenerator clas with a random seed of our choosing for reproducibility
    DataSet = DataGenerator(r_seed)

    #Doughnut and Sphere overlapping with various rotations on the doughnut
    rx_list = [0,0,45,0,45,45,45]
    ry_list = [45,0,0,45,0,45,45]
    rz_list = [0,45,0,45,45,0,45]

    for i in range(len(rx_list)):
        fig_num += 1 
        P, _ = DataSet.Doughnut(n,0,5.500001,0.5,0,0,0,0,0,0,0,0)
        _, Q = DataSet.Doughnut(0,m,0,0,0,0,0,4.5,0.5,0,0,0)
        P = DataSet.Rotate3D(P,rx_list[i],ry_list[i],rz_list[i],0,0,0)
        precision, recall = ComputePR(P,Q,k)
        p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
        i_precision, i_recall = ComputeIPR(P,Q,k)
        density, coverage = ComputeDC(P,Q,k)
        PlotData(P,Q,fig_num,7,6,2,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 7, 7, 2,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num

def Experiment14(r_seed, fig_num, k, n, m, C):
    '''
    Metrics run on Uniform data in the shape of a chessboard 
    '''

    #Initialize DataGenerator clas with a random seed of our choosing for reproducibility
    DataSet = DataGenerator(r_seed)

    #Parameters of chess board, x1 is xcoord of relative origin of board, y1 is y coord, and a is length of each square
    x1 = 0
    y1 = 0
    a  = 7

    #2D uniform chessboard
    fig_num += 1 
    P, Q = DataSet.UniformChessBoard(n,m,x1,y1,a)
    precision, recall = ComputePR(P,Q,num_clusters)
    p_cover, r_cover, P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn   = PRCover(P,Q,k,C)
    i_precision, i_recall = ComputeIPR(P,Q,k)
    density, coverage = ComputeDC(P,Q,k)
    PlotData(P,Q,fig_num,1,1,1,plotstyle='1d', save_fig='on',quick_time='on')
    PlotResults(precision,recall,i_precision,i_recall,density, coverage, p_cover, r_cover, k, C, fig_num, 1,1,1,save_fig='on',quick_time='on')
    PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)

    return fig_num

def MeasureConvergenceExperiment1(r_seed, num_samples, fig_num, C, C_0, start_val):
    '''
    Experiments to compare overlapping uniform distributions: 1 dimensional case and see metrics results vs true coverage
    '''

    #Initialize DataGenerator clas with a random seed of our choosing for reproducibility
    DataSet = DataGenerator(r_seed)

    fig_num += 1 

    #initializing properties for the measures
    num_clusters = 20
    
    #Fix distribution P, and move Q relative to P
    p_x1 = 0
    p_x2 = 10
    P_coords = np.array(([[p_x1],[p_x2]]))
    
    #distribution q position list 
    q_x1 = [2,3,4,5,6,7,8]
    q_x2 = [12,13,14,15,16,17,18]
    num_experiments = len(q_x1)

    #initialize arrays for measures
    I_precision = np.zeros(num_samples-start_val)
    I_recall = np.zeros(num_samples-start_val)
    c_precision = np.zeros(num_samples-start_val)
    c_recall = np.zeros(num_samples-start_val)
    density = np.zeros(num_samples-start_val)
    coverage = np.zeros(num_samples-start_val)

    #List of distribution Q's position 
    for i in range(num_experiments):
        for j in range(num_samples-start_val):

            #compute k 
            n = j + start_val
            m = n
            k = math.log(n) + C_0
            k = int(k+1)

            P, Q = DataSet.UniformData1D(n,m,p_x1,p_x2,q_x1[i],q_x2[i])

            #to ensure same choice of k for all         
            k_p = int(k/C+1)

            #compute measures
            c_precision[j], c_recall[j], P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn  = PRCover(P,Q,k_p,C)
            I_precision[j], I_recall[j] = ComputeIPR(P,Q,k)
            density[j], coverage[j] = ComputeDC(P,Q,k)

        #Compute the generative measures' values with maximum number of samples 
        I_precision_max = I_precision[num_samples-start_val-1]
        I_recall_max = I_recall[num_samples-start_val-1] 
        density_max = density[num_samples-start_val-1]
        coverage_max = coverage[num_samples-start_val-1]
        c_precision_max = c_precision[num_samples-start_val-1]
        c_recall_max = c_recall[num_samples-start_val-1]
        precision, recall = ComputePR(P,Q,num_clusters)

        Q_coords = np.array(([[q_x1[i]],[q_x2[i]]]))

        true_precision, true_recall = ComputeTruePR(P_coords, Q_coords)

        #Plot the data, results and manifold (for pr cover) for the measure values with max number of samples
        PlotData(P,Q,fig_num,0,0,0,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,I_precision_max,I_recall_max,density_max, coverage_max, c_precision_max, c_recall_max, k, C, fig_num, 0,0,0,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)
        
        #Plot precision and recall measures as a function of number of samples 
        PlotPrecisionConvergence(num_samples,true_precision, I_precision, density, c_precision, fig_num, 0, start_val, save_fig='on',quick_time='on')
        PlotRecallConvergence(num_samples,true_recall, I_recall, coverage, c_recall, fig_num, 0, start_val, save_fig='on',quick_time='on')
        fig_num += 1

    return fig_num 
    
def MeasureConvergenceExperiment2(r_seed, num_samples, fig_num, C, C_0, start_val):
    '''
    Experiments to compare overlapping uniform distributions: 2 dimensional case and see metrics results vs true coverage
    '''

    #Initialize DataGenerator class with a random seed of our choosing for reproducibility
    DataSet = DataGenerator(r_seed)

    fig_num += 1 

    #initializing properties for the measures
    num_clusters = 20

    #Fix distribution P, and move Q relative to P
    p_x1 = 0
    p_x2 = 10
    p_y1 = 0
    p_y2 = 10 
    P_coords = np.array(([[p_x1,p_y1],[p_x2,p_y2]]))
    
    #distribution q position list 
    q_x1 = [2,3,4,5,6,7,8]
    q_x2 = [12,13,14,15,16,17,18]
    q_y1 = [2,3,4,5,6,7,8]
    q_y2 = [12,13,14,15,16,17,18]
    num_experiments = len(q_x1)

    #initialize arrays for measures
    I_precision = np.zeros(num_samples-start_val)
    I_recall = np.zeros(num_samples-start_val)
    c_precision = np.zeros(num_samples-start_val)
    c_recall = np.zeros(num_samples-start_val)
    density = np.zeros(num_samples-start_val)
    coverage = np.zeros(num_samples-start_val)

    #List of distribution Q's position 
    for i in range(num_experiments):
        for j in range(num_samples-start_val):

            #compute k 
            n = j + start_val
            m = n
            k = math.log(n) + C_0
            k = int(k+1)

            P, Q = DataSet.UniformData2D(n,m,p_x1,p_x2,p_y1,p_y2,q_x1[i],q_x2[i],q_y1[i],q_y2[i])
        
            #to ensure same choice of k for all         
            k_p = int(k/C+1)
            
            #compute measures
            c_precision[j], c_recall[j], P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn  = PRCover(P,Q,k_p,C)
            I_precision[j], I_recall[j] = ComputeIPR(P,Q,k)
            density[j], coverage[j] = ComputeDC(P,Q,k)

        #Compute the generative measures' values with maximum number of samples 
        I_precision_max = I_precision[num_samples-start_val-1]
        I_recall_max = I_recall[num_samples-start_val-1] 
        density_max = density[num_samples-start_val-1]
        coverage_max = coverage[num_samples-start_val-1]
        c_precision_max = c_precision[num_samples-start_val-1]
        c_recall_max = c_recall[num_samples-start_val-1]
        precision, recall = ComputePR(P,Q,num_clusters)

        Q_coords = np.array(([[q_x1[i],q_y1[i]],[q_x2[i],q_y2[i]]]))

        true_precision, true_recall = ComputeTruePR(P_coords, Q_coords)

        #Plot the data, results and manifold (for pr cover) for the measure values with max number of samples
        PlotData(P,Q,fig_num,0,0,0,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,I_precision_max,I_recall_max,density_max, coverage_max, c_precision_max, c_recall_max, k, C, fig_num, 0,0,0,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)
        
        #Plot precision and recall measures as a function of number of samples 
        PlotPrecisionConvergence(num_samples,true_precision, I_precision, density, c_precision, fig_num, 1, start_val, save_fig='on',quick_time='on')
        PlotRecallConvergence(num_samples,true_recall, I_recall, coverage, c_recall, fig_num, 1,start_val, save_fig='on',quick_time='on')
        fig_num += 1

    return fig_num 
    
def MeasureConvergenceExperiment3(r_seed, num_samples, fig_num,C,C_0, start_val):
    '''
    Experiments to compare overlapping uniform distributions: 3 dimensional case and see metrics results vs true coverage
    '''

    #Initialize DataGenerator clas with a random seed of our choosing for reproducibility
    DataSet = DataGenerator(r_seed)

    fig_num += 1 

    #initializing properties for the measures
    num_clusters = 20
    
    #Fix distribution P, and move Q relative to P
    p_x1 = 0
    p_x2 = 10
    p_y1 = 0
    p_y2 = 10 
    p_z1 = 0
    p_z2 = 10
    P_coords = np.array(([[p_x1,p_y1,p_z1],[p_x2,p_y2,p_z2]]))
    
    #distribution q position list 
    q_x1 = [2,3,4,5,6,7,8]
    q_x2 = [12,13,14,15,16,17,18]
    q_y1 = [2,3,4,5,6,7,8]
    q_y2 = [12,13,14,15,16,17,18]
    q_z1 = [2,3,4,5,6,7,8]
    q_z2 = [12,13,14,15,16,17,18]
    num_experiments = len(q_x1)

    #initialize arrays for measures
    I_precision = np.zeros(num_samples-start_val)
    I_recall = np.zeros(num_samples-start_val)
    c_precision = np.zeros(num_samples-start_val)
    c_recall = np.zeros(num_samples-start_val)
    density = np.zeros(num_samples-start_val)
    coverage = np.zeros(num_samples-start_val)

    #List of distribution Q's position 
    for i in range(num_experiments):
        for j in range(num_samples-start_val):

            #compute k 
            n = j + start_val
            m = n
            k = math.log(n) + C_0
            k = int(k+1)

            P, Q = DataSet.UniformData3D(n,m,p_x1,p_x2,p_y1,p_y2,p_z1,p_z2,q_x1[i],q_x2[i],q_y1[i],q_y2[i],q_z1[i],q_z2[i])
        
            #to ensure same choice of k for all         
            k_p = int(k/C+1)
            
            #compute measures
            c_precision[j], c_recall[j], P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn  = PRCover(P,Q,k_p,C)
            I_precision[j], I_recall[j] = ComputeIPR(P,Q,k)
            density[j], coverage[j] = ComputeDC(P,Q,k)

        #Compute the generative measures' values with maximum number of samples 
        I_precision_max = I_precision[num_samples-start_val-1]
        I_recall_max = I_recall[num_samples-start_val-1] 
        density_max = density[num_samples-start_val-1]
        coverage_max = coverage[num_samples-start_val-1]
        c_precision_max = c_precision[num_samples-start_val-1]
        c_recall_max = c_recall[num_samples-start_val-1]
        precision, recall = ComputePR(P,Q,num_clusters)

        Q_coords = np.array(([[q_x1[i],q_y1[i],q_z1[i]],[q_x2[i],q_y2[i],q_z2[i]]]))

        true_precision, true_recall = ComputeTruePR(P_coords, Q_coords)

        #Plot the data, results and manifold (for pr cover) for the measure values with max number of samples
        PlotData(P,Q,fig_num,0,0,0,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,I_precision_max,I_recall_max,density_max, coverage_max, c_precision_max, c_recall_max, k, C, fig_num, 0,0,0,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)
        
        #Plot precision and recall measures as a function of number of samples 
        PlotPrecisionConvergence(num_samples,true_precision, I_precision, density, c_precision, fig_num, 2, start_val, save_fig='on',quick_time='on')
        PlotRecallConvergence(num_samples,true_recall, I_recall, coverage, c_recall, fig_num, 2, start_val, save_fig='on',quick_time='on')
        fig_num += 1

    return fig_num 

def MeasureConvergenceExperiment(r_seed, num_samples, fig_num,C,C_0, start_val, dimension):
    '''
    Experiments to compare overlapping uniform distributions: 3 dimensional case and see metrics results vs true coverage
    '''

    #Initialize DataGenerator clas with a random seed of our choosing for reproducibility
    DataSet = DataGenerator(r_seed)

    fig_num += 1 

    #initializing properties for the measures
    num_clusters = 20
    
    #Fix distribution P, and move Q relative to P
    init_coord = 0
    term_coord = 10
    
    #How much to move distribution Q by and how many times
    interval = 2
    num_steps = 10
    

    #initialize arrays for measures
    I_precision = np.zeros(num_samples-start_val)
    I_recall = np.zeros(num_samples-start_val)
    c_precision = np.zeros(num_samples-start_val)
    c_recall = np.zeros(num_samples-start_val)
    density = np.zeros(num_samples-start_val)
    coverage = np.zeros(num_samples-start_val)

    #List of distribution Q's position 
    for i in range(num_steps):
        for j in range(num_samples-start_val):

            #compute k 
            n = j + start_val
            m = n
            k = math.log(n) + C_0
            k = int(k+1)

            P, Q = DataSet.UniformPrismD(n,m,init_coord,term_coord,init_coord + i*interval, term_coord + i*interval,dimension)
        
            #to ensure same choice of k for all         
            k_p = int(k/C+1)
            
            #compute measures
            c_precision[j], c_recall[j], P_nQ_pts, P_nQ_knn, Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn  = PRCover(P,Q,k_p,C)
            I_precision[j], I_recall[j] = ComputeIPR(P,Q,k)
            density[j], coverage[j] = ComputeDC(P,Q,k)

        #Compute the generative measures' values with maximum number of samples 
        I_precision_max = I_precision[num_samples-start_val-1]
        I_recall_max = I_recall[num_samples-start_val-1] 
        density_max = density[num_samples-start_val-1]
        coverage_max = coverage[num_samples-start_val-1]
        c_precision_max = c_precision[num_samples-start_val-1]
        c_recall_max = c_recall[num_samples-start_val-1]
        precision, recall = ComputePR(P,Q,num_clusters)

        true_precision, true_recall = ComputeTruePR(P, Q)

        #Plot the data, results and manifold (for pr cover) for the measure values with max number of samples
        PlotData(P,Q,fig_num,0,0,0,plotstyle='1d', save_fig='on',quick_time='on')
        PlotResults(precision,recall,I_precision_max,I_recall_max,density_max, coverage_max, c_precision_max, c_recall_max, k, C, fig_num, 0,0,0,save_fig='on',quick_time='on')
        PlotManifolds(P,Q,P_nQ_pts,P_nQ_knn,Q_nP_pts, Q_nP_knn, PQ_pts, PQ_knn, (k*C),fig_num, plot_pts = True, save_fig = True,quick_time=True)
        
        #Plot precision and recall measures as a function of number of samples 
        PlotPrecisionConvergence(num_samples,true_precision, I_precision, density, c_precision, fig_num, 2, start_val, save_fig='on',quick_time='on')
        PlotRecallConvergence(num_samples,true_recall, I_recall, coverage, c_recall, fig_num, 2, start_val, save_fig='on',quick_time='on')
        fig_num += 1

    return fig_num 
    

