import numpy as np 
from math import pi, cos, sin, radians

class DataGenerator:
    def __init__(self, r_seed):
        '''
        Initialize random seed for reproducibility of experiments and data generation. 
        ''' 
        self.r_seed = r_seed

    def UniformData1D(self, n,m,a_P,b_P,a_Q,b_Q):
        '''
        Takes in parameters m: num gen samples, n: num real samples, a_P: start of uniform dist for P, b_P: 
        end of uniform dist for P, (a_Q and b_Q start end pts for Q) and a random seed value to generate 
        2 sets of samples P and Q for real and gen distributions respectively.
        '''
        np.random.seed(self.r_seed)

        P = np.random.uniform(a_P, b_P, (n,1))
        Q = np.random.uniform(a_Q, b_Q, (m,1))

        return P, Q 

    def UniformData2D(self,n, m, x1_P, x2_P, y1_P, y2_P, x1_Q, x2_Q, y1_Q, y2_Q):
        '''
        Takes values of endpoints to define 2 rectangles (for P and Q) a uniform distribution is then sampled
        over the defined rectangle. Also takes how many number of samples there are for each sample set.
        '''
        np.random.seed(self.r_seed)

        P_x = np.random.uniform(x1_P, x2_P, (n,1))
        P_y = np.random.uniform(y1_P, y2_P, (n,1))

        Q_x = np.random.uniform(x1_Q, x2_Q, (m,1))
        Q_y = np.random.uniform(y1_Q, y2_Q, (m,1))

        P = np.hstack([P_x, P_y])
        Q = np.hstack([Q_x, Q_y])

        return P, Q

    def UniformData3D(self,n, m, x1_P, x2_P, y1_P, y2_P, z1_P, z2_P, x1_Q, x2_Q, y1_Q, y2_Q, z1_Q, z2_Q):
        '''
        Takes values of endpoints to define 2 rectangular prisms (for P and Q) a uniform distribution is then sampled
        over respective x,y,z axes and then concatenated. Also takes how many number of samples there are for each sample set.
        '''
        np.random.seed(self.r_seed)

        P_x = np.random.uniform(x1_P, x2_P, (n,1))
        P_y = np.random.uniform(y1_P, y2_P, (n,1))
        P_z = np.random.uniform(z1_P, z2_P, (n,1))

        Q_x = np.random.uniform(x1_Q, x2_Q, (m,1))
        Q_y = np.random.uniform(y1_Q, y2_Q, (m,1))
        Q_z = np.random.uniform(z1_Q, z2_Q, (m,1))

        P = np.hstack([P_x, P_y, P_z])
        Q = np.hstack([Q_x, Q_y, Q_z])

        return P, Q

    def Gaussian2D(self,n,m, x_P, y_P, x_Q, y_Q, std_P, std_Q):
        '''
        Takes the num samples, mean (x,y coord separately) and std of each distribution (P and Q respectively)
        and returns a 2d normal distribution in particular the x and y coord of the true and gen dist.
        '''

        np.random.seed(self.r_seed)

        P = np.random.multivariate_normal(np.array([x_P,y_P]),np.array([[std_P,0],[0, std_P]]),(n))
        Q = np.random.multivariate_normal(np.array([x_Q,y_Q]),np.array([[std_Q,0],[0, std_Q]]),(m))

        return P, Q

    def Gaussian3D(self,n,m, x_P, y_P, z_P, x_Q, y_Q, z_Q, std_P, std_Q):
        '''
        Takes the num samples, mean (x,y coord separately) and std of each distribution (P and Q respectively)
        and returns a 3d normal distribution in particular the x and y coord of the true and gen dist.
        '''

        np.random.seed(self.r_seed)

        P = np.random.multivariate_normal(np.array([x_P,y_P,z_P]),np.array([[std_P,0,0],[0, std_P,0],[0,0,std_P]]),(n))
        Q = np.random.multivariate_normal(np.array([x_Q,y_Q,z_Q]),np.array([[std_Q,0,0],[0, std_Q,0],[0,0,std_Q]]),(m))

        return P, Q

    def Disc2D(self,n,m,P_r1,P_r2,P_xc,P_yc,Q_r1,Q_r2,Q_xc,Q_yc):
        '''
        Takes the num samples for both real and gen distribution, where r1 and r2 are the respective start and end radii
        of whichever distribution is being constructed. Points are generated uniformly between these 2 radii, also takes 
        coordinates for center for each disc (P_xc,P_yc) is (x,y) center of disc P.
        '''

        np.random.seed(self.r_seed)

        #Randomly sample from coords that define disc
        P_r = np.random.uniform(P_r1,P_r2,(n,1))
        P_theta = np.random.uniform(0,2*pi,(n,1))
        
        #Convert to xy coord and translate to desired center in xy coords
        P_x = (P_r * np.cos(P_theta)) + P_xc
        P_y = (P_r * np.sin(P_theta)) + P_yc

        #Repeat for Q dist with parameters provided for Q dist
        #********************************************

        Q_r = np.random.uniform(Q_r1,Q_r2,(m,1))
        Q_theta = np.random.uniform(0,2*pi,(m,1))
        
        Q_x = (Q_r * np.cos(Q_theta)) + Q_xc
        Q_y = (Q_r * np.sin(Q_theta)) + Q_yc

        P = np.hstack([P_x,P_y])
        Q = np.hstack([Q_x,Q_y])

        return P,Q

    def Sphere(self,n,m,P_r1,P_r2,P_xc,P_yc,P_zc,Q_r1,Q_r2,Q_xc,Q_yc,Q_zc):
        '''
        Takes the num samples for both real and gen distribution, where r1 and r2 are the respective start and end radii
        of whichever distribution is being constructed. Points are generated uniformly between these 2 radii, also takes 
        coordinates for center for each disc (P_xc,P_yc) is (x,y) center of disc P.
        '''

        np.random.seed(self.r_seed)

        #Randomly sample among the spherical coord that define the sphere
        P_r = np.random.uniform(P_r1,P_r2,(n,1))
        P_theta = np.random.uniform(0,2*pi,(n,1))
        P_phi   = np.random.uniform(0,pi,(n,1))

        #Convert from spherical to xyz and translate to desired centered xyz coords 
        P_x = (P_r * np.sin(P_phi)) * np.cos(P_theta) + P_xc
        P_y = (P_r * np.sin(P_phi)) * np.sin(P_theta) + P_yc
        P_z = (P_r * np.cos(P_phi)) + P_zc

        #Repeat for Q distribution with parameters provided for Q
        #************************************************************************

        Q_r = np.random.uniform(Q_r1,Q_r2,(m,1))
        Q_theta = np.random.uniform(0,2*pi,(m,1))
        Q_phi   = np.random.uniform(0,pi,(m,1))
        
        Q_x = (Q_r * np.sin(Q_phi)) * np.cos(Q_theta) + Q_xc
        Q_y = (Q_r * np.sin(Q_phi)) * np.sin(Q_theta) + Q_yc
        Q_z = (Q_r * np.cos(Q_phi)) + Q_zc

        P = np.hstack([P_x,P_y,P_z])
        Q = np.hstack([Q_x,Q_y,Q_z])

        return P,Q

    def Doughnut(self,n,m,P_r1,P_r2,P_xc,P_yc,P_zc,Q_r1,Q_r2,Q_xc,Q_yc,Q_zc):
        '''
        Takes the num samples for both real and gen dist respectively and creates a uniformly generated dist in the shape of a 
        doughnut. Takes the r1 which is radius from center of donut (center of main hole) to center of 'tube' center. Then the 
        'tube' is generated using r2, and of course the coordinates for the center of the doughnut are provided as well.
        Also key to note this is a doughnut lying flat on the z-axis, for rotated doughnuts please apply a transformation function.
        Also key note: THESE ARE NOT STANDARD SPHERICAL COORDINATES, read how they are defined. 
        '''

        np.random.seed(self.r_seed)

        #This radii is not the radius from the center of doughnut itself but radius from center of the 'tube'
        P_r = np.random.uniform(0,P_r2,(n,1))
        
        #P_theta is angle starting from x axis going all the way around in xy plane
        P_theta = np.random.uniform(0,2*pi,(n,1))

        #P_phi is angle that can be drawn out by r2, that is going 1 full circle maps out a cross section of the dougnut (a slice of the dougnut).        
        P_phi   = np.random.uniform(0,2*pi,(n,1))

        #Obtain x,y,z from radius and two angles that define doughnut, as well as translate among the x,y,z coord 
        P_x = (P_r1 + (P_r * np.cos(P_phi)))*np.cos(P_theta) + P_xc
        P_y = (P_r1 + (P_r * np.cos(P_phi)))*np.sin(P_theta) + P_yc
        P_z = (P_r * np.sin(P_phi)) + P_zc

        #Repeat for Q distribution with Q parameters
        #*******************************************

        Q_r = np.random.uniform(0,Q_r2,(m,1))
        Q_theta = np.random.uniform(0,2*pi,(m,1))
        Q_phi   = np.random.uniform(0,2*pi,(m,1))

        Q_x = (Q_r1 + (Q_r * np.cos(Q_phi)))*np.cos(Q_theta) + Q_xc
        Q_y = (Q_r1 + (Q_r * np.cos(Q_phi)))*np.sin(Q_theta) + Q_yc
        Q_z = (Q_r * np.sin(Q_phi)) + Q_zc

        P = np.hstack([P_x,P_y,P_z])
        Q = np.hstack([Q_x,Q_y,Q_z])   

        return P,Q
    
    def Rotate2D(self,data,theta,p_x,p_y):
        '''
        Rotates 2 dimensional data by theta degrees CCW w.r.t. the x axis, also the center coords of 
        the original object are needed so that the new object is rotated about the center of the original object
        instead of also being translated.
        '''

        #Extract the x and y coord
        x = data[:,0]
        y = data[:,1]
        x = x.reshape(len(x),1)
        y = y.reshape(len(y),1)

        #In order for the object to be rotated about center we translate to origin
        translate_matrix = np.hstack([x - p_x, y - p_y])

        #Now create the 2D rotation matrix
        rotation_matrix = np.array(([cos(radians(theta)), - sin(radians(theta)) ], [sin(radians(theta)), cos(radians(theta))]))

        #Obtain translated matrix about origin (and np gives transposed ans naturally)
        temp_matrix = np.matmul(rotation_matrix, np.transpose(translate_matrix))
        new_matrix = np.transpose(temp_matrix)

        #Extract new matrix's x and y components
        new_x = new_matrix[:,0]
        new_y = new_matrix[:,1]
        new_x = new_x.reshape(len(new_x),1)
        new_y = new_y.reshape(len(new_y),1)
        
        #Finally obtain the final result by translating back to the center of original object
        new_matrix = np.hstack([new_x+p_x, new_y+p_y])

        return new_matrix

    def Rotate3D(self, data, theta_x, theta_y, theta_z, p_x, p_y, p_z):
        '''
        Rotates 3 dimensional data by theta_p where p is the respective axis
        and theta is the amount of rotation in degrees counterclockwise from 
        the respective axis. Also the coords of the center of the original 
        object must be provided so that the rotation is about the center of the
        original object and the not the origin, so there is no translation as well. 
        '''

        #Convert all angles to radians
        theta_xr = radians(theta_x)
        theta_yr = radians(theta_y)
        theta_zr = radians(theta_z)

        #Extract the x,y, and z coord
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        x = x.reshape(len(x),1)
        y = y.reshape(len(y),1)
        z = z.reshape(len(z),1)

        #In order for the object to be rotated about center we translate to origin
        translate_matrix = np.hstack([x - p_x, y - p_y, z - p_z])
        
        #Obtain each individual axis' rotation matrix
        R_x = np.array(([1,0,0],[0,cos(theta_xr),-sin(theta_xr)],[0,sin(theta_xr),cos(theta_xr)]))
        R_y = np.array(([cos(theta_yr),0,sin(theta_yr)],[0,1,0],[-sin(theta_yr),0,cos(theta_yr)]))
        R_z = np.array(([cos(theta_zr),-sin(theta_zr),0],[sin(theta_zr),cos(theta_zr),0],[0,0,1]))


        #Obtain total rotation matrix by definition R = RzRyRx
        R_matrix = np.matmul(R_z,(np.matmul(R_y,R_x)))

        #Finally rotate the chosen data
        temp_matrix = np.matmul(R_matrix, np.transpose(translate_matrix))
        new_matrix = np.transpose(temp_matrix)
        
        #Extract new matrix's x and y components
        new_x = new_matrix[:,0]
        new_y = new_matrix[:,1]
        new_z = new_matrix[:,2]
        new_x = new_x.reshape(len(new_x),1)
        new_y = new_y.reshape(len(new_y),1)
        new_z = new_z.reshape(len(new_z),1)

        #Combine the xyz components for final result        
        new_matrix = np.hstack([new_x + p_x,new_y + p_y,new_z + p_z])

        return new_matrix
