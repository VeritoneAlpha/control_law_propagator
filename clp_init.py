#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:09:32 2018
Init file for holding arguments and functions to be called by the control law propagator (clp).

"""
import numpy as np
import scipy as sp
import ode


def H_T_p(q,p,u):
    # for q-dot
    return np.ones(np.shape(q))+q

def H_T_q(q,p,u):
    # for p-dot
    return np.ones(np.shape(p))+p
    
def Q_u(q,p,u):
    # for u-dot
    return np.ones(np.shape(u))+u
    

# Inputs for numerical propagator
q_0 = np.array([0,2])
p_0 = np.array([1,4])
u_0 = np.array([2])
qpu_vec = np.hstack([q_0, p_0, u_0])
state_dim = 2
Gamma = 1

# Inputs for numerical integration
integrateTol = 10**-3
integrateMaxIter = 40

# Inputs for sliding window
t_0 = 0
T =  4
K=2

# inputs for sliding window 
t_terminal = 100
