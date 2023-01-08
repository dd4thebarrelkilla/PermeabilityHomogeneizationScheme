# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 10:00:58 2022

@author: Daniel
"""
import numpy as np
import matplotlib.pyplot as plt
import Perm_fun as pf


def PermCalc(L, steps, parameters ,check = False):      
    
    init = L / steps
    
    
    # Initial guess for flow fron position of each layer
    x0 = np.array([init , init])
    
    
    # Imposed flowfronts along the part
    x_avg = np.linspace(init, L, steps)
    
    
    # Record permeability along the part
    perm = np.zeros(steps)
    
    
    # Record the flowfront position of each layer along the part
    stepfront = np.zeros([len(x0), steps], dtype = np.float64)
    
    
    # Initialise parameters
    h, H, Kxx, Kt, P_in, eta = pf.getParameters(x0, parameters)
    
    
    #Calculate KTF (perm) for each imposed flowfront
    for i in range (steps):    
        
        x = pf.NR_method(x0, x_avg[i], parameters,RELAX = 0.5)
        
        
        #equation 4
        perm[i] = (x_avg[i]/H) * sum(Kxx * h / x)
    
        stepfront[:,i] = x
        
        x0 = x
        
    
        
    KAA_avg = sum(Kxx*h)/sum(h)
    
    
    KTF_avg = np.average(perm)
    
    ratio = KAA_avg / perm
    
    if check == True:
        print('\n KTF = ', KTF_avg,
              '\n KAA = ', KAA_avg)
    
    return perm, KAA_avg, x_avg, H, ratio


L = 1
steps = 1000

Kxx1 = 1e-9
Kxx2 = 5e-11                      
Kt = [Kxx1*0.1, Kxx1*0.01, Kxx1*0.001]

S = [0.1, 0.01, 0.001]
D = ['-bx', '-ro', '-y1']

for i in range(len(Kt)):
    parameters = np.array([1e-9, 5e-11, Kt[i]])
    # Plot ratio of KAA and KTF over normalized flowfront
    perm, KAA_avg, x_avg, H, ratio = PermCalc(L, steps, parameters)
    plt.plot(x_avg/H, ratio, str(D[i]),label = 'Kx1/Kt = ' + str(S[i]))
    
plt.xlabel('Normalized Avg. Flow Front position x_avg/H')
plt.ylabel('KAA/<KTF>')
plt.grid()
plt.ylim(1, 1.7)
plt.legend()
plt.grid()



