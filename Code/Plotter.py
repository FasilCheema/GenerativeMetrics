#from turtle import color
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle

def PlotData(P,Q, fig_num, distP_val, distQ_val, overlap_val, plotstyle = '1d', save_fig = 'off',quick_time='off'):
    # Takes the samples and plots them depending on the dimensionality
    
    type_dist    = ['1D Uniform','2D Uniform','3D Uniform','2D Gaussian','3D Gaussian','2D Uniform Disc','Spherical','Doughnut']
    type_overlap = [' matching distributions',' disjoint distributions',' overlapping distributions'] 
    
    dim_P = P.shape[1]
    dim_Q = Q.shape[1]

    n = P.shape[0]
    m = Q.shape[0]
    
    if (dim_P == 1) and (dim_Q == 1):
        # Code for standard 1D Histograms
        if plotstyle != '3d':
            # Setup and customize figure and plot
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot()
            ax.set_xlabel('value')
            ax.set_ylabel('frequency')
            ax.set_title('Histogram of P (true data) and Q (gen data) both are '+type_dist[distP_val]+type_overlap[overlap_val])

            ax.hist(P, bins = 'auto', color='blue', alpha=0.5, label=('P True distribution (n=%d)'%(n)))
            ax.hist(Q, bins = 'auto', color='red' , alpha=0.5, label=('Q Gen  distribution (m=%d)'%(m)))
            ax.legend([('True distribution P (n=%d)'%(n)),('Generated distribution Q (m=%d)'%(m))])
            plt.legend()

            # Saves an image of the plot in the appropriate directory with appropriate naming.
            if save_fig == 'on':
                #fig.savefig("Experiments/InputData%d.png"%(fig_num))
                fig.savefig("TestExperiments/InputData%d.png"%(fig_num))

            #If in a hurry just saves plots without displaying
            if quick_time == 'off':
                plt.show()
            else:
            #to save memory closes figures for mass deployment of figures
                fig.clear()
                plt.close(fig)
        else:
            # Code to do 3d histograms
            fig = plt.figure(figsize=(10,10))
            ax  = fig.add_subplot(111, projection='3d')

            hist, bins = np.histogram(P, bins='auto')
            xs = (bins[:-1] + bins[1:])/2
            ax.bar(xs, hist, zs = 0, alpha=0.8, color='blue')

            hist, bins = np.histogram(Q, bins='auto')
            xs = (bins[:-1] + bins[1:])/2
            ax.bar(xs, hist, zs =  10, alpha=0.8, color ='red')
            
            # Saves an image of the plot in the appropriate directory with appropriate naming.
            if save_fig == 'on':
                #fig.savefig("Experiments/InputData%d.png"%(fig_num))
                fig.savefig("TestExperiments/InputData%d.png"%(fig_num))

    elif dim_P == 2:
        # assumes 2d plots
        P_x = P[:,0]
        P_y = P[:,1]
        Q_x = Q[:,0]
        Q_y = Q[:,1]

        fig, ax = plt.subplots(figsize=(10,10))
        ax.scatter(P_x,P_y, color = 'blue')
        ax.scatter(Q_x,Q_y, color = 'red')
        ax.legend([('True Distribution P (n=%d)'%(n)), ('Gen Distribution Q (m=%d)'%(m))])
        ax.set_title('Plotting 2d experiment of generated ' +type_dist[distQ_val]+ ' and real '+type_dist[distP_val]+type_overlap[overlap_val])
        ax.set_ylabel('y axis')
        ax.set_xlabel('x axis')
        
        # Saves an image of the plot in the appropriate directory with appropriate naming.
        if save_fig == 'on':
                #fig.savefig("Experiments/InputData%d.png"%(fig_num))
                fig.savefig("TestExperiments/InputData%d.png"%(fig_num))
        
        #If in a hurry just saves plots without displaying
        if quick_time == 'off':
            plt.show()
        else:
            #to save memory closes figures for mass deployment of figures
            fig.clear()
            plt.close(fig)

    elif dim_P == 3:
        # assumes 3d plots
        P_x = P[:,0]
        P_y = P[:,1]
        P_z = P[:,2]
        Q_x = Q[:,0]
        Q_y = Q[:,1]
        Q_z = Q[:,2]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(P_x,P_y,P_z, color = 'blue')
        ax.scatter(Q_x,Q_y,Q_z, color = 'red')
        ax.legend([('True Distribution P (n=%d)'%(n)), ('Gen Distribution Q (m=%d)'%(m))])
        ax.set_title('Plotting 3D experiment of generated '+type_dist[distQ_val]+' and real '+type_dist[distP_val]+type_overlap[overlap_val])
        ax.set_ylabel('y axis')
        ax.set_xlabel('x axis')
        
        # Saves an image of the plot in the appropriate directory with appropriate naming.
        if save_fig == 'on':
                #fig.savefig("Experiments/InputData%d.png"%(fig_num))
                fig.savefig("TestExperiments/InputData%d.png"%(fig_num))
        
        #If in a hurry just saves plots without displaying
        if quick_time == 'off':
            plt.show()
        else:
        #to save memory closes figures for mass deployment of figures
            fig.clear()
            plt.close(fig)
    else:
        print('Not plotting above 3 dimensions.')

def PlotResults(precision, recall, I_precision, I_recall, density, coverage, c_precision, c_recall, k, C, fig_num, distP_val, distQ_val, overlap_val, save_fig ='off', quick_time='off'):

    #Preformatting of text
    type_dist    = ['1D Uniform','2D Uniform','3D Uniform','2D Gaussian','3D Gaussian','2D Uniform Disc','Spherical','Doughnut']
    type_overlap = [' matching distributions',' disjoint distributions',' overlapping distributions'] 
    
    #Plots precision and recall
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.fill_between(recall, 0, precision, color='green')
    ax.set_title('Precision and Recall of real '+type_dist[distP_val]+' and gen '+type_dist[distQ_val]+type_overlap[overlap_val])
    ax.set_xlabel(r'Recall $ \beta $')
    ax.set_ylabel(r'Precision $ \alpha $')

    #Displays values of metric scores
    ax.text(0.65, 1.09, r'Density = %4.2f , Coverage = %4.2f' % (density, coverage), fontsize=12)
    ax.text(0.65, 1.05, r'I_precision = %4.2f , I_recall = %4.2f' % (I_precision, I_recall), fontsize=12)
    ax.text(0.65, 1.13, r'C_precision = %4.2f , C_recall = %4.2f' % (c_precision, c_recall), fontsize=12)

    #Shows what values of k are used
    ax.text(0.05, -0.09, r"k = %d for D&C and Improved P&R and k' = %d, k = %d for Cover P&R" % (k,C*k,k), fontsize=14)

    #Saves an image of the plot in the appropriate directory with appropriate naming.
    if save_fig == 'on':
        #fig.savefig("Experiments/Results%d.png"%(fig_num))
        fig.savefig("TestExperiments/Results%d.png"%(fig_num))
    
    #If in a hurry does not display plots just saves
    if quick_time == 'off':
        plt.show()
    else:
    #to save memory closes figures for mass deployment of figures
        fig.clear()
        plt.close(fig)

def PlotManifolds(P,Q,P_disjoint_Q_pts,P_disjoint_Q_knn,Q_disjoint_P_pts, Q_disjoint_P_knn, joint_supp_pts, joint_supp_knn, k,fig_num, plot_pts = False, save_fig = True,quick_time=True):
    #Define spheres by radii
    #find k'-nn and use this
    num_P = P.shape[0]
    num_Q = Q.shape[0]

    #Assumes dimension of P and Q are the same, so only 1 is needed
    dim_P = P.shape[1]
    dim_Q = Q.shape[1]


    if (dim_P == 1) and (dim_Q == 1):
        #For 1D data we plot on a 2d surface which is why we set y axis to be a constant of 1
        P_y = np.ones((num_P,1))
        Q_y = np.ones((num_Q,1))

        fig = plt.figure(figsize=(10,10))
        ax  = fig.add_subplot()
        ax.set_xlabel('value')
        ax.set_ylabel('1-Dimensional Data')
        ax.set_title('Histogram of P (true data) and Q (gen data) with PR Cover metric manifolds')
        ax.set_ylim([-5, 5])

        #plot points from P and Q
        if plot_pts == True:
            ax.scatter(P,P_y, color ='blue')
            ax.scatter(Q,Q_y, color ='red')

        #for each 1D point we add a patch that shows the 1D 'ball' or rectangle in this case that is the knn ball for each sample point
        if type(P_disjoint_Q_pts) == int:
            print('no points for P that are disjoint from Q')
        else:            
            for i in range(P_disjoint_Q_pts.shape[0]):
                ax.add_patch(Rectangle((P_disjoint_Q_pts[i][0]-P_disjoint_Q_knn[i][0],-1),P_disjoint_Q_knn[i][0]*2,2,color='blue',alpha=0.1))

        if type(Q_disjoint_P_pts) == int:
            print('no points for Q that are disjoint from P')
        else:            
            for j in range(Q_disjoint_P_pts.shape[0]):
                ax.add_patch(Rectangle((Q_disjoint_P_pts[j][0]-Q_disjoint_P_knn[j][0],-1),Q_disjoint_P_knn[j][0]*2,2,color='red',alpha=0.1))

        if type(joint_supp_pts) == int:
            print('no joint support!')
        else:
            for k in range(joint_supp_pts.shape[0]):
                ax.add_patch(Rectangle((joint_supp_pts[k][0]-joint_supp_knn[k][0],-1),joint_supp_knn[k][0]*2,2,color='green',alpha=0.1))

        #Adjusting legend settings 
        leg = ax.legend(labels = ["Only P","Only Q","Joint support"])
        leg.legendHandles[0].set_color('blue')
        leg.legendHandles[1].set_color('red')
        leg.legendHandles[2].set_color('green')
        leg.legendHandles[0].set_alpha(1)
        leg.legendHandles[1].set_alpha(1)
        leg.legendHandles[2].set_alpha(1)

        #Checking if we are saving the figure 
        if save_fig == True:
            #fig.savefig("Experiments/PRCover_Manifold%d.png"%(fig_num))
            fig.savefig("TestExperiments/PRCover_Manifold%d.png"%(fig_num))

        #If in a rush we do not display the image
        if quick_time == False:
            plt.show()

    elif (dim_P == 2) and (dim_Q == 2):
        P_x = P[:,0]
        P_y = P[:,1]

        Q_x = Q[:,0]
        Q_y = Q[:,1]

        fig = plt.figure(figsize=(10,10))
        ax  = fig.add_subplot()
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_title('Plot of P (true data) and Q (gen data) with PR Cover metric manifolds')

        #plot points from P and Q
        if plot_pts == True:
            ax.scatter(P_x,P_y, color ='blue')
            ax.scatter(Q_x,Q_y, color ='red')

        #for each 2D point we add a patch that shows the 1D 'ball' or circle in this case that is the knn ball for each sample point
        if type(P_disjoint_Q_pts) == int:
            print('no points for P that are disjoint from Q')
        else:            
            for i in range(P_disjoint_Q_pts.shape[0]):
                ax.add_patch(Circle((P_disjoint_Q_pts[i][0],P_disjoint_Q_pts[i][1]),radius = P_disjoint_Q_knn[i][0],color='blue',alpha=0.1))
            
        if type(Q_disjoint_P_pts) == int:
            print('no points for Q that are disjoint from P')
        else:            
            for j in range(Q_disjoint_P_pts.shape[0]):
                ax.add_patch(Circle((Q_disjoint_P_pts[j][0],Q_disjoint_P_pts[j][1]),radius = Q_disjoint_P_knn[j][0],color='red',alpha=0.1))
        
        if type(joint_supp_pts) == int:
            print('no joint support')
        else:
            for k in range(joint_supp_pts.shape[0]):
                ax.add_patch(Circle((joint_supp_pts[k][0],joint_supp_pts[k][1]),radius = joint_supp_knn[k][0],color='green',alpha=0.1))
        
        #Adjusting legend settings 
        leg = ax.legend(labels = ["Only P","Only Q","Joint support"])
        leg.legendHandles[0].set_color('blue')
        leg.legendHandles[1].set_color('red')
        leg.legendHandles[2].set_color('green')
        leg.legendHandles[0].set_alpha(1)
        leg.legendHandles[1].set_alpha(1)
        leg.legendHandles[2].set_alpha(1)

        #Checking if we are saving the figure 
        if save_fig == True:
            #fig.savefig("Experiments/PRCover_Manifold%d.png"%(fig_num))
            fig.savefig("TestExperiments/PRCover_Manifold%d.png"%(fig_num))

        #If in a rush we do not display the image
        if quick_time == False:
            plt.show()
    
    elif (dim_P == 3) and (dim_Q == 3):
        P_x = P[:,0]
        P_y = P[:,1]
        P_z = P[:,2]

        Q_x = Q[:,0]
        Q_y = Q[:,1]
        Q_z = Q[:,2]

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        fig = plt.figure(figsize=(10,10))
        ax  = fig.add_subplot(111,projection='3d')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_title('Plot of P (true data) and Q (gen data) with PR Cover metric manifolds')

        #plot points from P and Q
        if plot_pts == True:
            ax.scatter(P_x,P_y,P_z, color ='blue')
            ax.scatter(Q_x,Q_y,Q_z, color ='red')

        #for each 3D point we add a patch that shows the 3D 'ball' or sphere in this case that is the knn ball for each sample point
        if type(P_disjoint_Q_pts) == int:
            print('no points for P that are disjoint from Q')
        else:            
            for i in range(P_disjoint_Q_pts.shape[0]):
                ax.plot_surface(P_disjoint_Q_knn[i][0] * np.outer(np.cos(u),np.sin(v)) + P_disjoint_Q_pts[i][0],P_disjoint_Q_knn[i][0] * np.outer(np.sin(u),np.sin(v)) + P_disjoint_Q_pts[i][1],P_disjoint_Q_knn[i][0] * np.outer(np.ones(np.size(u)),np.cos(v)) + P_disjoint_Q_pts[i][2],color='blue',alpha=0.1)

        if type(Q_disjoint_P_pts) == int:
            print('no points for Q that are disjoint from P')
        else:            
            for j in range(Q_disjoint_P_pts.shape[0]):
                ax.plot_surface(Q_disjoint_P_knn[j][0] * np.outer(np.cos(u),np.sin(v)) + Q_disjoint_P_pts[j][0],Q_disjoint_P_knn[j][0] * np.outer(np.sin(u),np.sin(v)) + Q_disjoint_P_pts[j][1],Q_disjoint_P_knn[j][0] * np.outer(np.ones(np.size(u)),np.cos(v)) + Q_disjoint_P_pts[j][2],color='red',alpha=0.1)

        if type(joint_supp_pts) == int:
            print('no joint support!')
        else:            
            for k in range(joint_supp_pts.shape[0]):
                ax.plot_surface(joint_supp_knn[k][0] * np.outer(np.cos(u),np.sin(v)) + joint_supp_pts[k][0],joint_supp_knn[k][0] * np.outer(np.sin(u),np.sin(v)) + joint_supp_pts[k][1],joint_supp_knn[k][0] * np.outer(np.ones(np.size(u)),np.cos(v)) + joint_supp_pts[k][2],color='green',alpha=0.1)
        
        #leg = ax.legend(labels = ["Only P","Only Q","Joint support"])
        #leg.legendHandles[0].set_color('blue')
        #leg.legendHandles[1].set_color('red')
        #leg.legendHandles[2].set_color('green')
        #leg.legendHandles[0].set_alpha(1)
        #leg.legendHandles[1].set_alpha(1)
        #leg.legendHandles[2].set_alpha(1)

        #Checking if we are saving the figure 
        if save_fig == True:
            #fig.savefig("Experiments/PRCover_Manifold%d.png"%(fig_num))
            fig.savefig("TestExperiments/PRCover_Manifold%d.png"%(fig_num))
    
        #If in a rush we do not display the image
        if quick_time == False: 
            plt.show()
    else:
        print('Dimension Error')
        
