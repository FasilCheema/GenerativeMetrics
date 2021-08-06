'''
Author: Fasil Cheema
Purpose: Attempting to showcase examples on which common generative model evaluation measures fail. 
'''

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from math import sin,cos,radians

#Initial parameters 
radius = 1 
epsilon = 0.1

#Real samples, and angles to define the generated samples with respect to real samples
x_vals = [1,1,1,2,3,3,3,2]
y_vals = [1,2,3,3,3,2,1,1]
a_list = np.linspace(225,-90,8) #list of angles starting from bottom corner point going clockwise

x = np.array(x_vals)
y = np.array(y_vals)

#Initialize the three arrays the boundary values (on the circle), the overrepresented generated values, and the underrepresented values.
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

#This code section is for plots
fig = plt.figure()

#Define Circles
circle1 = plt.Circle((1,1),radius,color='b',fill = False,clip_on=False)
circle2 = plt.Circle((1,2),radius,color='b',fill = False,clip_on=False)
circle3 = plt.Circle((1,3),radius,color='b',fill = False,clip_on=False)
circle4 = plt.Circle((2,1),radius,color='b',fill = False,clip_on=False)
circle5 = plt.Circle((2,2),radius,color='b',fill = False,clip_on=False)
circle6 = plt.Circle((2,3),radius,color='b',fill = False,clip_on=False)
circle7 = plt.Circle((3,1),radius,color='b',fill = False,clip_on=False)
circle8 = plt.Circle((3,2),radius,color='b',fill = False,clip_on=False)
circle9 = plt.Circle((3,3),radius,color='b',fill = False,clip_on=False)

#Plot all generated and real samples as well as circles
plt.figure(0)
ax1 = plt.gca()
ax1.scatter(x,y,color='brown')
ax1.scatter(xg1,yg1,color='green')
ax1.scatter(xg2,yg2,color='red')
ax1.add_patch(circle1)
ax1.add_patch(circle2)
ax1.add_patch(circle3)
ax1.add_patch(circle4)
ax1.add_patch(circle5)
ax1.add_patch(circle6)
ax1.add_patch(circle7)
ax1.add_patch(circle8)
ax1.add_patch(circle9)

plt.figure(1)

plt.show()