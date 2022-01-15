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
                fig.savefig("Experiments/InputData%d.png"%(fig_num))

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
                fig.savefig("Experiments/InputData%d.png"%(fig_num))

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
                fig.savefig("Experiments/InputData%d.png"%(fig_num))
        
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
                fig.savefig("Experiments/InputData%d.png"%(fig_num))
        
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
        fig.savefig("Experiments/Results%d.png"%(fig_num))
    
    #If in a hurry does not display plots just saves
    if quick_time == 'off':
        plt.show()
    else:
    #to save memory closes figures for mass deployment of figures
        fig.clear()
        plt.close(fig)

def Plot_Manifolds(P,Q,dist_P,dist_Q,ind_P,ind_Q,k,C):
    #Define spheres by radii
    #find k'-nn and use this
    num_P = P.shape[0]
    num_Q = Q.shape[0]

    #Assumes dimension of P and Q are the same, so only 1 is needed
    dim_P = P.shape[1]
    dim_Q = Q.shape[1]

    if (dim_P == 1) and (dim_Q == 1):
        P_y = np.zeros((num_P,1))
        Q_y = np.zeros((num_Q,1))

        fig = plt.figure(figsize=(10,10))
        ax  = fig.add_subplot()
        ax.set_xlabel('value')
        ax.set_ylabel('1-Dimensional Data')
        ax.set_title('Histogram of P (true data) and Q (gen data) with PR Cover metric manifolds')
    
        ax.scatter(P,P_y, color ='blue')
        ax.scatter(Q,Q_y, color ='red')

        for i in range(num_P):
        
        #ax.add_patch(Rectangle(x,y),width,height, color='green',alpha=0.3)

    elif (dim_P == 2) and (dim_Q == 2):
        
    elif (dim_P == 3) and (dim_Q == 3):

    else:
        print('Dimension Error')
        
