# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:49:44 2022

@author: Daniel

    Equations are referencing to the paper:
    Effective permeability averaging scheme to address in-plane anisotropy
    effects in multi-layered preforms - S.P. Bancora

"""
import numpy as np
import sys
# import timeit, os
# import matplotlib.pyplot as plt
# import pdb
# import warnings


def getParameters(x, parameters):
    
    #These are the material properties used
    
    nx = len(x)
    h1 = 0.006                         #Ply thickness (m)
    h = np.ones(nx)*h1                 #Thickness vector
    H = np.sum(h)                      #Preform thickness (m)
    
    # Kxx1 = 1e-9                        #Permeability of UD[1]
    # Kxx2 = 5e-11                       #Permeability of UD[2]
    # #Kxx3 = 1e-7                       #Permeability of BIAX
    # Kxx = np.array([Kxx1, Kxx2])       #permeability vector
    # Kt = Kxx1*0.1                      #avg. transverse permeability
    
    # P_in = 1                           #Inlet pressure (Pa)
    # eta = 1                            #Resin viscosity (Pa.s)
    
    Kxx1 = parameters[0]               #Permeability of UD[1]
    Kxx2 = parameters[1]               #Permeability of UD[2]
    Kxx = np.array([Kxx1, Kxx2])       #permeability vector
    Kt = parameters[2]                 #avg. transverse permeability
    
    P_in = 1                           #Inlet pressure (Pa)
    eta = 1                            #Resin viscosity (Pa.s)
    
    return h, H, Kxx, Kt, P_in, eta



def NormalFlowrate(viscosity, Kxx, P_in, l, h):
    
    #Find normal flow rate
    #Equation 11
    
    flow_rate_normal = (Kxx * P_in * h)/(viscosity * l)
    
    return flow_rate_normal




def TransvFlowrate(viscosity, Kt, P_in, l1, l2, h1, h2):
    
    #Find transverse flow rate
    #Equation 12
    
    flow_rate_transv = (Kt * P_in * (l1 - l2)**2)/(viscosity * l1 * (h1 + h2))

    
    if l1 < l2:
        
        flow_rate_transv = -flow_rate_transv    
        
    return flow_rate_transv
    


def FlowFront(x, x_avg):
    
    #Relation between average flow front position and the one of each layer.
    #Must be = 0.
    #Equation 13
    
    f = (np.sum(x)/len(x)) - x_avg
    
    return f
       

    
def NetFlowrate(x, parameters):
    
    #Equation 7

    nx = len(x)
    N = len(x) - 1    


    #Parameters
    h, H, Kxx, Kt, P_in, eta = getParameters(x, parameters)


 
    Q_N = np.zeros(nx)
    
    #Get first and last value of Q_N
    Q_N[0] = (NormalFlowrate(eta, Kxx[0], P_in, x[0], h[0]) 
            - TransvFlowrate(eta, Kt, P_in, x[0], x[1], h[0], h[1]))
    
    
    Q_N[N] = (NormalFlowrate(eta, Kxx[N], P_in, x[N], h[N]) 
            + TransvFlowrate(eta, Kt, P_in, x[N-1], x[N], h[N-1], h[N]))
    
    
    # #Get intermediate values of Q_N
    if N > 2 :
        
        for i in range(1,N-1):
            
            
            Q_N[i] = (NormalFlowrate(eta, Kxx[i], P_in, x[i], h[i]) 
                      - TransvFlowrate(eta, Kt, P_in, x[i], x[i+1], h[i], h[i+1]) 
                      + TransvFlowrate(eta, Kt, P_in, x[i-1], x[i], h[i-1], h[i]))
            
        
    return Q_N


def Q_comp(x, x_avg, parameters):   
     
    #Find the difference between the volumetric flow rates.
    #Must be = 0.
    #Relates to equation 6
    
    nx = len(x)
    N = len(x) - 1
    

    Q_N = NetFlowrate(x, parameters)
    
    
    Q_fun = np.zeros(nx)
    
    
    for i in range(N):
        
        Q_fun[i] = Q_N[i] - Q_N[i+1]
        
        
    #last equation, flow front relationship
    Q_fun[N] = FlowFront(x, x_avg)
    
    return Q_fun


def Jacobian(x, function, x_avg, parameters):
    
    #Used in the Newton Raphson method

    eps = 1e-6
    
    nx = len(x)
    J = np.zeros([nx, nx], dtype = np.float64)
    f = function
    
    
    for i in range(nx):
        
        EPS = np.zeros(nx) 
        EPS[i] = eps
        
        
        J[:, i] = 0.5 * (f(x + EPS, x_avg, parameters) - f(x -  EPS, x_avg, parameters))/eps
            
    detJ = np.linalg.det(J)
            
    if abs(detJ) < 1e-12:
        print('\n Warning ! det(J) = 0 !',
              '\n Change the RELAXATION FACTOR')
        
        sys.exit()
        
        #pdb.set_trace()
    
    return J


def NR_method(x, x_avg, parameters,check = False, RELAX = 1):
    
    #Newton Raphson method for the iterative method


    #SHOOTING METHOD INITIALISATION
    # RELAX = 0.5   #between 0 and 1
    iter = 0
    converged = 0
    Tol = 1e-06
    maxiter = 1e3
    if check == True:        
        print('\n Quasi Newton method iterations started')
    
    function = Q_comp
        
    while converged == 0 and iter < maxiter:
        
        x_OLD = x
        
        fun = Q_comp(x, x_avg, parameters)
        residual = np.max(abs(fun))
        
        if residual < Tol:
            if check == True:
                print('\n Converged! root = ', x,' residual =', residual)
            
            converged = 1
                        
        else: 
            if check == True:
                print('\n Root = ', x,' Iteration =', iter,' residual =', residual)
            iter += 1
            
            J = Jacobian(x, function, x_avg, parameters)
            
            x = x - np.linalg.solve(J, fun)
            
            #Use relaxation to faciliate convergence of nonlinear system
            x = x_OLD*(1 - RELAX) + x*RELAX
        
        if iter > maxiter:
            
            print('\n Warning ! Reached too many iterations',
                  'Check RELAXATION FACTOR')
            
            sys.exit()
    

    return x
