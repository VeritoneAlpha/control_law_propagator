# Jordan Makansi
# 12/19/18

import numerical_propagator as prp

import unittest   
import numpy as np 
import abc
import scipy as sp
import ode

from blackboard import *

class batteryAgent:
    
    def __init__(self, blackboard, state_indices, control_indices, q_s_0=None, p_l_0=None, p_mf_0=None, u_s_0=None, q_s_dot=None, gamma=1, Gamma=1, name='', integrateTol=10**-5, integrateMaxIter=400, t_0=0, T=2, K=4, t_terminal=4, n_s=10):
        ''' state_indices (list of integers): This list tells which states pertain to this agent. e.g. [1,2] would 
        tell us that states 1 and 2 pertain to this agent.
        
        A word on notation:  The notation used for the methods of the agent is:  
            - If it is a partial derivative: <denominator>_rhs_H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g., 
            "qp_rhs_H_l_u" denotes the partial derivative with respect to q and p of the terms in the local Hamiltonian that contain control variables.
            - If it is a hamiltonian: H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g. "H_mf_nou" denotes the mean field hamiltonian with terms not containing u.
        '''
        self.state_indices = state_indices
        self.control_indices = control_indices
        self.bb = blackboard
 

    def L_l(self, q_1, q_B, q_1_dot, q_B_dot, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        '''
        Inputs:
            states:
                q_1: charge 1 at time t
                q_B: charge of battery at time t
            control:
                u_B: control of battery
            data:
                q_1_0:
                q_B_0:
                v_c_u_0:
                v_c_1_0:
            parameters:
                c_1:
                R_0:
                R_1:
                v_a:
                Q_0:
                beta:
                v_N:
        '''
        # TODO: replace as a function of self.K and self.T
        delta = 1

        V_c_1 = (c_1/2.0)*((((q_1 - q_1_0)/c_1) + v_c_1_0)**2 - V_c_1_0)
        V_c_u = 0.5*(u_B*(-(q_B - q_B_0))**2+2*(-(q_B - q_B_0)*V_c_u_0))
        D_R_0 = 0.5*(-q_B_dot**2)*R_0*delta  
        D_R_1 = 0.5*((-q_B_dot - q_1_dot)**2)*R_1*delta
        F_B = -(v_N/beta**2)*(beta*q_B - beta*q_B_0 + (Q_0*beta - Q_0)*np.log( (Q_0 - Q_0*beta + beta*q_B) / (Q_0 - Q_0*beta + beta*q_B_0)))
        F_B_out = -(q_B - q_B_0)*v_a

        L_l = -V_c_1 - V_c_u + D_R_0 + D_R_1 + F_B - F_B_out
        return  L_l
        

    def 
