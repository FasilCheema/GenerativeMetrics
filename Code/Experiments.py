def Experiments():
# Set of experiments to be conducted
    r_seed = 7
    k = 3
    
    # 1D point generator 
    #************************ 
     
    # Case 1, matching 1D dist
    k1 = k
    fig_num = 1
    P1,Q1 = UniformData1D(1000,1000,0,10,0,10,r_seed)
    PlotData(P1,Q1,fig_num, plotstyle='1d',save_fig = 'on')
    c1_precision, c1_recall = PRCover(P1,Q1,k1)
    density1, coverage1 = ComputeDC(P1,Q1,k1)
    p1, r1 = ComputePR(P1,Q1,k1)
    Ip1, Ir1 = ComputeIPR(P1,Q1,k1)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')

    # Case 2, disjoint 1D dist
    k2 = k
    fig_num = 2
    P2,Q2 = UniformData1D(1000,1000,0,10,11,20,r_seed)
    PlotData(P2,Q2,fig_num,plotstyle='1d',save_fig='on')
    c2_precision, c2_recall = PRCover(P2,Q2,k2)
    density2, coverage2 = ComputeDC(P2,Q2,k2)
    p2, r2 = ComputePR(P2,Q2,k2)
    Ip2, Ir2 = ComputeIPR(P2,Q2,k2)
    PlotResults(p2,r2,Ip2,Ir2,density2,coverage2,c2_precision,c2_recall,fig_num,save_fig='on')

    # Case 3, overlapping 1D dist
    k3 = k
    fig_num = 3
    P3,Q3 = UniformData1D(1000,1000,0,10,5,15,r_seed)
    PlotData(P3,Q3,fig_num,plotstyle='1d',save_fig='on')
    c3_precision, c3_recall = PRCover(P3,Q3,k3)
    density3, coverage3 = ComputeDC(P3,Q3,k3)
    p3, r3 = ComputePR(P3,Q3,k3)
    Ip3, Ir3 = ComputeIPR(P3,Q3,k3)
    PlotResults(p3,r3,Ip3,Ir3,density3,coverage3,c3_precision,c3_recall,fig_num,save_fig='on')

    # Case 4, matching 2D dist
    k4 = k
    fig_num = 4
    P4, Q4 = UniformData2D(1000,1000,5,13,7,19,5,13,7,19,r_seed)
    PlotData(P4,Q4,fig_num,plotstyle='1d',save_fig='on')
    c4_precision, c4_recall = PRCover(P4,Q4,k4)
    density4, coverage4 = ComputeDC(P4,Q4,k4)
    p4, r4 = ComputePR(P4,Q4,k4)
    Ip4, Ir4 = ComputeIPR(P4,Q4,k4)
    PlotResults(p4,r4,Ip4,Ir4,density4,coverage4,c4_precision,c4_recall,fig_num,save_fig='on')

    # Case 5, disjoint 2D dist
    k5 = k
    fig_num = 5
    P5, Q5 = UniformData2D(1000,1000,5,7,13,19,23,29,37,49,r_seed)
    PlotData(P5,Q5,fig_num,plotstyle='1d',save_fig='on')
    c5_precision, c5_recall = PRCover(P5,Q5,k5)
    density5, coverage5 = ComputeDC(P5,Q5,k5)
    p5, r5 = ComputePR(P5,Q5,k5)
    Ip5, Ir5 = ComputeIPR(P5,Q5,k5)
    PlotResults(p5,r5,Ip5,Ir5,density5,coverage5,c5_precision,c5_recall,fig_num,save_fig='on')
     
    # Case 6, overlapping 2D dist
    k6 = k
    fig_num = 6
    P6, Q6 = UniformData2D(1000,1000,5,15,25,35,10,20,30,40,r_seed)
    PlotData(P6,Q6,fig_num,plotstyle='1d',save_fig='on')
    c6_precision, c6_recall = PRCover(P6,Q6,k6)
    density6, coverage6 = ComputeDC(P6,Q6,k6)
    p6, r6 = ComputePR(P6,Q6,k6)
    Ip6, Ir6 = ComputeIPR(P6,Q6,k6)
    PlotResults(p6,r6,Ip6,Ir6,density6,coverage6,c6_precision,c6_recall,fig_num,save_fig='on')

    # Case 7, matching Gaussians
    k7 = k
    fig_num = 7
    P7, Q7 = Gaussian2D(1000,1000,17,23,17,23,1,1,7)
    PlotData(P7,Q7,fig_num,plotstyle='1d',save_fig='on')
    c7_precision, c7_recall = PRCover(P7,Q7,k7)
    density7, coverage7 = ComputeDC(P7,Q7,k7)
    p7, r7 = ComputePR(P7,Q7,k7)
    Ip7, Ir7 = ComputeIPR(P7,Q7,k7)
    PlotResults(p7,r7,Ip7,Ir7,density7,coverage7,c7_precision,c7_recall,fig_num,save_fig='on')

    # Case 8, 'disjoint' Gaussians
    k8 = k
    fig_num = 8
    P8, Q8 = Gaussian2D(1000,1000,17,23,117,123,1,1,7)
    PlotData(P8,Q8,fig_num,plotstyle='1d',save_fig='on')
    c8_precision, c8_recall = PRCover(P8,Q8,k8)
    density8, coverage8 = ComputeDC(P8,Q8,k8)
    p8, r8 = ComputePR(P8,Q8,k8)
    Ip8, Ir8 = ComputeIPR(P8,Q8,k8)
    PlotResults(p8,r8,Ip8,Ir8,density8,coverage8,c8_precision,c8_recall,fig_num,save_fig='on')

    # Case 9, 'overlapping' Gaussians
    k9 = k
    fig_num = 9
    P9, Q9 = Gaussian2D(1000,1000,20,20,21,21,1,1,7)
    PlotData(P9,Q9,fig_num,plotstyle='1d',save_fig='on')
    c9_precision, c9_recall = PRCover(P9,Q9,k9)
    density9, coverage9 = ComputeDC(P9,Q9,k9)
    p9, r9 = ComputePR(P9,Q9,k9)
    Ip9, Ir9 = ComputeIPR(P9,Q9,k9)
    PlotResults(p9,r9,Ip9,Ir9,density9,coverage9,c9_precision,c9_recall,fig_num,save_fig='on')

def Experiments2():
# Set of experiments to be conducted swapped P and Q 
    r_seed = 7
    k = 3
    
    #1D point generator 
    #************************ 
     
    # Case 1, matching 1D dist
    k1 = k
    fig_num = 10
    Q1,P1 = UniformData1D(1000,1000,0,10,0,10,r_seed)
    PlotData(P1,Q1,fig_num, plotstyle='1d',save_fig = 'on')
    c1_precision, c1_recall = PRCover(P1,Q1,k1)
    density1, coverage1 = ComputeDC(P1,Q1,k1)
    p1, r1 = ComputePR(P1,Q1,k1)
    Ip1, Ir1 = ComputeIPR(P1,Q1,k1)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')

    # Case 2, disjoint 1D dist
    k2 = k
    fig_num = 11
    Q2,P2 = UniformData1D(1000,1000,0,10,11,20,r_seed)
    PlotData(P2,Q2,fig_num,plotstyle='1d',save_fig='on')
    c2_precision, c2_recall = PRCover(P2,Q2,k2)
    density2, coverage2 = ComputeDC(P2,Q2,k2)
    p2, r2 = ComputePR(P2,Q2,k2)
    Ip2, Ir2 = ComputeIPR(P2,Q2,k2)
    PlotResults(p2,r2,Ip2,Ir2,density2,coverage2,c2_precision,c2_recall,fig_num,save_fig='on')

    # Case 3, overlapping 1D dist
    k3 = k
    fig_num = 12
    Q3,P3 = UniformData1D(1000,1000,0,10,5,15,r_seed)
    PlotData(P3,Q3,fig_num,plotstyle='1d',save_fig='on')
    c3_precision, c3_recall = PRCover(P3,Q3,k3)
    density3, coverage3 = ComputeDC(P3,Q3,k3)
    p3, r3 = ComputePR(P3,Q3,k3)
    Ip3, Ir3 = ComputeIPR(P3,Q3,k3)
    PlotResults(p3,r3,Ip3,Ir3,density3,coverage3,c3_precision,c3_recall,fig_num,save_fig='on')

    # 2D Uniform point generator
    # *********************************

    # Case 4, matching 2D dist
    k4 = k
    fig_num = 13
    Q4, P4 = UniformData2D(1000,1000,5,13,7,19,5,13,7,19,r_seed)
    PlotData(P4,Q4,fig_num,plotstyle='1d',save_fig='on')
    c4_precision, c4_recall = PRCover(P4,Q4,k4)
    density4, coverage4 = ComputeDC(P4,Q4,k4)
    p4, r4 = ComputePR(P4,Q4,k4)
    Ip4, Ir4 = ComputeIPR(P4,Q4,k4)
    PlotResults(p4,r4,Ip4,Ir4,density4,coverage4,c4_precision,c4_recall,fig_num,save_fig='on')

    # Case 5, disjoint 2D dist
    k5 = k
    fig_num = 14
    Q5, P5 = UniformData2D(1000,1000,5,7,13,19,23,29,37,49,r_seed)
    PlotData(P5,Q5,fig_num,plotstyle='1d',save_fig='on')
    c5_precision, c5_recall = PRCover(P5,Q5,k5)
    density5, coverage5 = ComputeDC(P5,Q5,k5)
    p5, r5 = ComputePR(P5,Q5,k5)
    Ip5, Ir5 = ComputeIPR(P5,Q5,k5)
    PlotResults(p5,r5,Ip5,Ir5,density5,coverage5,c5_precision,c5_recall,fig_num,save_fig='on')
     
    # Case 6, overlapping 2D dist
    k6 = k
    fig_num = 15
    Q6, P6 = UniformData2D(1000,1000,5,15,25,35,10,20,30,40,r_seed)
    PlotData(P6,Q6,fig_num,plotstyle='1d',save_fig='on')
    c6_precision, c6_recall = PRCover(P6,Q6,k6)
    density6, coverage6 = ComputeDC(P6,Q6,k6)
    p6, r6 = ComputePR(P6,Q6,k6)
    Ip6, Ir6 = ComputeIPR(P6,Q6,k6)
    PlotResults(p6,r6,Ip6,Ir6,density6,coverage6,c6_precision,c6_recall,fig_num,save_fig='on')
    # 2D Gaussian Generator
    # *************************

    # Case 7, matching Gaussians
    k7 = k
    fig_num = 16
    Q7, P7 = Gaussian2D(1000,1000,17,23,17,23,1,1,7)
    PlotData(P7,Q7,fig_num,plotstyle='1d',save_fig='on')
    c7_precision, c7_recall = PRCover(P7,Q7,k7)
    density7, coverage7 = ComputeDC(P7,Q7,k7)
    p7, r7 = ComputePR(P7,Q7,k7)
    Ip7, Ir7 = ComputeIPR(P7,Q7,k7)
    PlotResults(p7,r7,Ip7,Ir7,density7,coverage7,c7_precision,c7_recall,fig_num,save_fig='on')

    # Case 8, 'disjoint' Gaussians
    k8 = k
    fig_num = 17
    Q8, P8 = Gaussian2D(1000,1000,17,23,117,123,1,1,7)
    PlotData(P8,Q8,fig_num,plotstyle='1d',save_fig='on')
    c8_precision, c8_recall = PRCover(P8,Q8,k8)
    density8, coverage8 = ComputeDC(P8,Q8,k8)
    p8, r8 = ComputePR(P8,Q8,k8)
    Ip8, Ir8 = ComputeIPR(P8,Q8,k8)
    PlotResults(p8,r8,Ip8,Ir8,density8,coverage8,c8_precision,c8_recall,fig_num,save_fig='on')

    # Case 9, 'overlapping' Gaussians
    k9 = k
    fig_num = 18
    Q9, P9 = Gaussian2D(1000,1000,20,20,21,21,1,1,7)
    PlotData(P9,Q9,fig_num,plotstyle='1d',save_fig='on')
    c9_precision, c9_recall = PRCover(P9,Q9,k9)
    density9, coverage9 = ComputeDC(P9,Q9,k9)
    p9, r9 = ComputePR(P9,Q9,k9)
    Ip9, Ir9 = ComputeIPR(P9,Q9,k9)
    PlotResults(p9,r9,Ip9,Ir9,density9,coverage9,c9_precision,c9_recall,fig_num,save_fig='on')

def Experiments3():
# Set of experiments to be conducted
    r_seed = 7
    k = 3
    
    # Case 7, matching Gaussians
    k7 = k
    fig_num = 19
    P7, Q7 = Gaussian3D(1000,1000,17,23,29,17,23,29,1,1,7)
    PlotData(P7,Q7,fig_num,plotstyle='1d',save_fig='on')
    c7_precision, c7_recall = PRCover(P7,Q7,k7)
    density7, coverage7 = ComputeDC(P7,Q7,k7)
    p7, r7 = ComputePR(P7,Q7,k7)
    Ip7, Ir7 = ComputeIPR(P7,Q7,k7)
    PlotResults(p7,r7,Ip7,Ir7,density7,coverage7,c7_precision,c7_recall,fig_num,save_fig='on')

    # Case 8, 'disjoint' Gaussians
    k8 = k
    fig_num = 20
    P8, Q8 = Gaussian3D(1000,1000,17,23,29,117,123,129,1,1,7)
    PlotData(P8,Q8,fig_num,plotstyle='1d',save_fig='on')
    c8_precision, c8_recall = PRCover(P8,Q8,k8)
    density8, coverage8 = ComputeDC(P8,Q8,k8)
    p8, r8 = ComputePR(P8,Q8,k8)
    Ip8, Ir8 = ComputeIPR(P8,Q8,k8)
    PlotResults(p8,r8,Ip8,Ir8,density8,coverage8,c8_precision,c8_recall,fig_num,save_fig='on')

    # Case 9, 'overlapping' Gaussians
    k9 = k
    fig_num = 21
    P9, Q9 = Gaussian3D(1000,1000,20,20,20,21,21,21,1,1,7)
    PlotData(P9,Q9,fig_num,plotstyle='1d',save_fig='on')
    c9_precision, c9_recall = PRCover(P9,Q9,k9)
    density9, coverage9 = ComputeDC(P9,Q9,k9)
    p9, r9 = ComputePR(P9,Q9,k9)
    Ip9, Ir9 = ComputeIPR(P9,Q9,k9)
    PlotResults(p9,r9,Ip9,Ir9,density9,coverage9,c9_precision,c9_recall,fig_num,save_fig='on')

def Experiments4():
    # 3D Uniform Data + 3D Gaussian Data
    k = 3
    fig_num = 22
    P1,   _ = Gaussian3D(1000,1000,0,0,0,0,0,0,1,1,7)
    Q1_x, _ = UniformData1D(10000,1000,-1,1,-1,1,7)
    Q1_y, _ = UniformData1D(10000,1000,-1,1,-1,1,8)
    Q1_z, _ = UniformData1D(10000,1000,-1,1,-1,1,9)
    Q1 = np.hstack((Q1_x,Q1_y,Q1_z))
    PlotData(P1,Q1,fig_num,plotstyle='1d',save_fig='on')
    c1_precision, c1_recall = PRCover(P1,Q1,k)
    density1, coverage1 = ComputeDC(P1,Q1,k)
    p1,r1 = ComputePR(P1,Q1,k)
    Ip1,Ir1 = ComputeIPR(P1,Q1,k)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')

def Experiments5():
    # 3D Uniform Data + 3D Gaussian Data
    k = 5
    fig_num = 23
    P1,   _ = Gaussian3D(1000,1000,0,0,0,0,0,0,1,1,7)
    Q1_x, _ = UniformData1D(10000,1000,-1,1,-1,1,7)
    Q1_y, _ = UniformData1D(10000,1000,-1,1,-1,1,8)
    Q1_z, _ = UniformData1D(10000,1000,-1,1,-1,1,9)
    Q1 = np.hstack((Q1_x,Q1_y,Q1_z))
    PlotData(P1,Q1,fig_num,plotstyle='1d',save_fig='on')
    c1_precision, c1_recall = PRCover(P1,Q1,k)
    density1, coverage1 = ComputeDC(P1,Q1,k)
    p1,r1 = ComputePR(P1,Q1,k)
    Ip1,Ir1 = ComputeIPR(P1,Q1,k)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')

def Experiments6():
    # 3D Uniform Data + 3D Gaussian Data
    k = 10
    fig_num = 24
    P1,   _ = Gaussian3D(1000,1000,0,0,0,0,0,0,1,1,7)
    Q1_x, _ = UniformData1D(10000,1000,-1,1,-1,1,7)
    Q1_y, _ = UniformData1D(10000,1000,-1,1,-1,1,8)
    Q1_z, _ = UniformData1D(10000,1000,-1,1,-1,1,9)
    Q1 = np.hstack((Q1_x,Q1_y,Q1_z))
    PlotData(P1,Q1,fig_num,plotstyle='1d',save_fig='on')
    c1_precision, c1_recall = PRCover(P1,Q1,k)
    density1, coverage1 = ComputeDC(P1,Q1,k)
    p1,r1 = ComputePR(P1,Q1,k)
    Ip1,Ir1 = ComputeIPR(P1,Q1,k)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')

def Experiments7():
    # 3D Uniform Data + 3D Gaussian Data
    k = 5
    fig_num = 25
    P1,   _ = Gaussian3D(1000,1000,0,0,0,0,0,0,1,1,7)
    Q1_x, _ = UniformData1D(1000,1000,-1,1,-1,1,7)
    Q1_y, _ = UniformData1D(1000,1000,-1,1,-1,1,8)
    Q1_z, _ = UniformData1D(1000,1000,-1,1,-1,1,9)
    Q1 = np.hstack((Q1_x,Q1_y,Q1_z))
    PlotData(P1,Q1,fig_num,plotstyle='1d',save_fig='on')
    c1_precision, c1_recall = PRCover(P1,Q1,k)
    density1, coverage1 = ComputeDC(P1,Q1,k)
    p1,r1 = ComputePR(P1,Q1,k)
    Ip1,Ir1 = ComputeIPR(P1,Q1,k)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')

def Experiments8():
    # 3D Uniform Data + 3D Gaussian Data
    k = 5
    fig_num = 26
    P1,   _ = Gaussian3D(5000,1000,0,0,0,0,0,0,1,1,7)
    Q1_x, _ = UniformData1D(1000,1000,-1,1,-1,1,7)
    Q1_y, _ = UniformData1D(1000,1000,-1,1,-1,1,8)
    Q1_z, _ = UniformData1D(1000,1000,-1,1,-1,1,9)
    Q1 = np.hstack((Q1_x,Q1_y,Q1_z))
    PlotData(P1,Q1,fig_num,plotstyle='1d',save_fig='on')
    c1_precision, c1_recall = PRCover(P1,Q1,k)
    density1, coverage1 = ComputeDC(P1,Q1,k)
    p1,r1 = ComputePR(P1,Q1,k)
    Ip1,Ir1 = ComputeIPR(P1,Q1,k)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')

def Experiments9():
    #Doughnut + 3D Gaussian
    k = 3
    fig_num = 27
    P1, _   = Gaussian3D(1000,1000,0,0,0,0,0,0,1,1,7)
    _, Q1   = Doughnut(1000,1000,1,0.5,0,0,0,1,0.5,0,0,0,7)
    PlotData(P1,Q1,fig_num,plotstyle='1d',save_fig='on')
    c1_precision, c1_recall = PRCover(P1,Q1,k)
    density1, coverage1 = ComputeDC(P1,Q1,k)
    p1,r1 = ComputePR(P1,Q1,k)
    Ip1,Ir1 = ComputeIPR(P1,Q1,k)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')

def Experiments10():
    #Doughnut + 3D Gaussian
    k = 3
    fig_num = 28
    P1, _   = Gaussian3D(2000,2000,0,0,0,0,0,0,1,1,7)
    _, Q1   = Doughnut(2000,2000,5,1,0,0,0,5,1,0,0,0,7)
    PlotData(P1,Q1,fig_num,plotstyle='1d',save_fig='on')
    c1_precision, c1_recall = PRCover(P1,Q1,k)
    density1, coverage1 = ComputeDC(P1,Q1,k)
    p1,r1 = ComputePR(P1,Q1,k)
    Ip1,Ir1 = ComputeIPR(P1,Q1,k)
    PlotResults(p1,r1,Ip1,Ir1,density1,coverage1,c1_precision,c1_recall,fig_num,save_fig='on')
