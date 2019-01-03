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
            And p_rhs_H_mf_u denotes the partial derivative with respect to p of the terms in the mean field Hamiltonian of the terms containing "u".
            - If it is a hamiltonian: H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g. "H_mf_nou" denotes the mean field hamiltonian with terms not containing u.
        '''
        self.state_indices = state_indices
        self.state_dim = len(self.state_indices)
        self.control_indices = control_indices
        self.control_dim = len(self.control_indices)
        self.bb = blackboard
        self.gamma = gamma
 
    # local methods
    def L_l(self, q_1, q_B, q_1_dot, q_B_dot, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
    #def L_l(self, q_s, q_s_dot, u_s, **kwargs) :
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
        # q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N = kwargs['q_1_0'], kwargs['q_B_0'], kwargs['v_c_u_0'], kwargs['v_c_1_0'],kwargs['c_1'], kwargs['R_0'], kwargs['R_1'], kwargs['v_a'], kwargs['Q_0'], kwargs['beta'],kwargs['v_N']

        V_c_1 = (c_1/2.0)*((((q_1 - q_1_0)/c_1) + v_c_1_0)**2 - v_c_1_0)
        V_c_u = 0.5*(u_B*(-(q_B - q_B_0))**2+2*(-(q_B - q_B_0)*v_c_u_0))
        D_R_0 = 0.5*(-q_B_dot**2)*R_0*delta  
        D_R_1 = 0.5*((-q_B_dot - q_1_dot)**2)*R_1*delta
        F_B = -(v_N/beta**2)*(beta*q_B - beta*q_B_0 + (Q_0*beta - Q_0)*np.log( (Q_0 - Q_0*beta + beta*q_B) / (Q_0 - Q_0*beta + beta*q_B_0)))
        F_B_out = -(q_B - q_B_0)*v_a

        L_l = -V_c_1 - V_c_u + D_R_0 + D_R_1 + F_B - F_B_out
        return  L_l


    def H_l(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        H_l_nou = self.H_l_nou(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        H_l = H_l_nou + np.dot(H_l_nou, u_B)
        return H_l

    def H_l_nou(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # TODO: replace as a function of self.K and self.T
        delta = 1
        term_1 = 0.5*(((p_B-p_1)**2)/(R_0 * delta)) + 0.5*((p_1**2)/(R_1*delta))
        term_2 = (c_1/2)*((((q_1-q_1_0)/c_1) + v_c_1_0)**2 - v_c_1_0**2) 
        term_3 = (-(q_B - q_B_0)*v_c_u_0) 
        term_4 =  (v_N/beta)*(beta*q_B - beta*q_B_0 +(Q_0*beta - Q_0)*np.log( (Q_0 - Q_0*beta + beta*q_B) / (Q_0 - Q_0*beta + beta*q_B_0)))
        term_5 = -(q_B-q_B_0)*v_a
        return term_1 + term_2 + term_3 + term_4 + term_5

    def H_l_u(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        term_1 = 0.5*(-(q_B - q_B_0))**2 
        return term_1

    def q_rhs_H_l_nou(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # TODO: replace as a function of self.K and self.T
        delta = 1
        q_1_rhs_H_l_nou = (q_1 - q_1_0)/c_1 + v_c_1_0
        q_B_rhs_H_l_nou = -v_c_u_0 - v_a + (v_N/beta) + (v_N*Q_0*(beta-1))/(beta*(Q_0 - Q_0*beta + beta*q_B))
        q_rhs_H_l_nou = np.concatenate([np.array([q_1_rhs_H_l_nou]), np.array([q_B_rhs_H_l_nou])])
        return q_rhs_H_l_nou

    def p_rhs_H_l_nou(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # TODO: replace as a function of self.K and self.T
        delta = 1
        p_1_rhs_H_l_nou = -p_B/(R_0*delta) + (p_1/delta)*((1/R_0)+(1/R_1))
        p_B_rhs_H_l_nou = (p_B - p_1)/(R_0*delta)
        p_rhs_H_l_nou = np.concatenate([np.array([p_1_rhs_H_l_nou]), np.array([p_B_rhs_H_l_nou])])
        return p_rhs_H_l_nou

    def q_rhs_H_l_u(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # TODO: replace as a function of self.K and self.T
        delta = 1
        q_1_rhs_H_l_u = 0
        q_B_rhs_H_l_u = q_B - q_B_0
        q_rhs_H_l_u = np.concatenate([np.array([q_1_rhs_H_l_u]), np.array([q_B_rhs_H_l_u])])
        return np.array([q_rhs_H_l_u])

    def p_rhs_H_l_u(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # TODO: replace as a function of self.K and self.T
        delta = 1
        p_1_rhs_H_l_u = 0
        p_B_rhs_H_l_u = 0
        p_rhs_H_l_u = np.concatenate([np.array([p_1_rhs_H_l_u]), np.array([p_B_rhs_H_l_u])])
        return np.array([p_rhs_H_l_u])

    def q_rhs_H_l(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        q_rhs_H_l = self.q_rhs_H_l_nou(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N) + np.dot(self.q_rhs_H_l_u(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N).T, np.array([u_B]))
        # should return something of dimension state_dim 
        return q_rhs_H_l

    def p_rhs_H_l(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        p_rhs_H_l = self.p_rhs_H_l_nou(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N) + np.dot(self.p_rhs_H_l_u(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N).T, np.array([u_B]))
        # should return something of dimension state_dim 
        return p_rhs_H_l

    def qp_rhs_H_l(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        q_rhs_H_l = self.q_rhs_H_l( q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        p_rhs_H_l = self.p_rhs_H_l(q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        return np.concatenate([q_rhs_H_l, p_rhs_H_l])

    # Mean Field methods
    def H_mf_nou(self, q_mf, p_mf, u_mf):
    #...other methods for mean field 
        pass

    def q_rhs_H_mf_u(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # should return 2D numpy array of dimension control_dim x state_dim
        # TODO: Change to actual mean field.  For now just use local functions.
        p_H_mf_u_dot_1 = self.p_rhs_H_l_u(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        # Normally, we would wrap this in a numpy array like np.array([p_H_mf_u_dot_1]), but since we are stealing from local in this case it is not necessary
        return p_H_mf_u_dot_1

    def p_rhs_H_mf_u(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # should return 2D numpy array of dimension control_dim x state_dim
        # TODO: Change to actual mean field.  For now just use local functions.
        q_H_mf_u_dot_1 = self.q_rhs_H_l_u(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        return q_H_mf_u_dot_1

    def q_rhs_H_mf_nou(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # should return 1D numpy array of dimension 1 x state_dim
        # TODO: Change to actual mean field.  For now just use local functions.
        p_H_mf_u_dot_1 = self.p_rhs_H_l_nou(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        return p_H_mf_u_dot_1

    def p_rhs_H_mf_nou(self, q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # should return 1D numpy array of dimension 1 x state_dim
        # TODO: Change to actual mean field.  For now just use local functions.
        q_H_mf_u_dot_1 = self.q_rhs_H_l_nou(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        return q_H_mf_u_dot_1

    def p_rhs_H_mf(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
        p_rhs_H_mf_u = self.p_rhs_H_mf_u(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        assert np.shape(p_rhs_H_mf_u)==(len(self.control_indices), self.state_dim) # first dimension should be number of controls, inner dimension should be state_dim
        # since only one control, no need for dot product here
        p_rhs_H_mf_u_summed = np.dot(self.p_rhs_H_mf_u(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N).T, np.array([u_B]))        
        return self.p_rhs_H_mf_nou(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N) + p_rhs_H_mf_u_summed

    def q_rhs_H_mf(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
        q_rhs_H_mf_u = self.q_rhs_H_mf_u(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        assert np.shape(q_rhs_H_mf_u)==(len(self.control_indices), self.state_dim) # first dimension should be number of controls, inner dimension should be state_dim
        q_rhs_H_mf_u_summed = np.dot(self.q_rhs_H_mf_u(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N).T, np.array([u_B]))        
        return self.q_rhs_H_mf_nou(q_1, q_B, p_1, p_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N) + q_rhs_H_mf_u_summed

    def qp_rhs_H_mf(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        # remember that we want to propagate as much as possible together in the same rhs function for numerical purposes
        # remember that q_rhs here is w.r.t p_mf but p_rhs here is w.r.t q_s
        q_H_mf_dot = self.p_rhs_H_mf(q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        p_H_mf_dot = self.q_rhs_H_mf(q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        return np.concatenate([q_H_mf_dot, p_H_mf_dot])

    def qp_rhs(self, t, **kwargs): 
    #if all inputs needed explicitly use def qp_rhs(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N):
        q_1 = kwargs['q_1']
        q_B = kwargs['q_B']        
        p_1 = kwargs['p_1']       
        p_B = kwargs['p_B']       
        u_B =kwargs['u_B']        
        q_1_0_0 = kwargs['q_1_0']       
        q_B_0 = kwargs['q_B_0']       
        v_c_u_0 = kwargs['v_c_u_0']       
        v_c_1_0 = kwargs['v_c_1_0']       
        c_1 = kwargs['c_1']       
        R_0 = kwargs['R_0']       
        R_1 = kwargs['R_1']       
        v_a = kwargs['v_a']       
        Q_0 = kwargs['Q_0']       
        beta = kwargs['beta']       
        v_N = kwargs['v_N']       
 
        state_dim = self.state_dim
 
        qp_rhs_H_mf = self.qp_rhs_H_mf(q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)

        q_rhs_H_mf = qp_rhs_H_mf[:state_dim]
        p_rhs_H_mf = qp_rhs_H_mf[state_dim:]

        qp_rhs_H_l = self.qp_rhs_H_l(q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N)
        q_rhs_H_l = qp_rhs_H_l[:state_dim]
        p_rhs_H_l = qp_rhs_H_l[state_dim:]

        q_s_dot = self.gamma*p_rhs_H_mf + (1-self.gamma)*p_rhs_H_l
        p_mf_dot = q_rhs_H_mf
        p_l_dot = -1*q_rhs_H_l
        
        return np.concatenate([q_s_dot, p_l_dot, p_mf_dot])
 

    def u_rhs(self, t, u_vec, **kwargs):
        u_s = kwargs['u_0']
        state_dim = kwargs['state_dim']
        q_mf_dot = kwargs['q_mf_dot']
        q_s_dot = kwargs['q_s_dot']
        p_l_dot = kwargs['p_l_dot']
        p_mf_dot = kwargs['p_mf_dot']
        q_mf = kwargs['q_mf']
        u_mf = kwargs['u_mf']
        qp_vec = kwargs['qp_vec']
        H_l_D = kwargs['H_l_D']
        Beta_mf = kwargs['Beta_mf']
        Beta_l = kwargs['Beta_l']
        alpha_mf = kwargs['alpha_mf']
        alpha_l = kwargs['alpha_l']
        q_s = qp_vec[:state_dim]
        p_l = qp_vec[state_dim:2*state_dim]
        p_mf = qp_vec[2*state_dim:]
        u_s_dot = np.array([])
        for j in range(self.control_dim):
            '''for each control, we need to:
                1) Compute and get a 1D np.array for each of alpha_l_j, etc.
                2) Compute u_s_dot_j = -1*self.Gamma*(self.gamma*(alpha_mf_j + np.dot(Beta_mf_j, u_s)) + (1-self.gamma)*(alpha_l_j + np.dot(Beta_l_j,u_s)))
                3) Concatenate all of the u_s_dot_j to construct u_s_dot in a 1D np.array
            '''
            
            Beta_mf_j,Beta_l_j = Beta_mf[j], Beta_l[j] 
            alpha_mf_j, alpha_l_j = alpha_mf[j], alpha_l[j]
            # Beta_mf_j, Beta_l_j should be vectors
            # alpha_mf_j, alpha_l_j should be scalars
            u_s_dot_j = -1*self.Gamma*(self.gamma*(alpha_mf_j + np.dot(Beta_mf_j, u_s)) + (1-self.gamma)*(alpha_l_j + np.dot(Beta_l_j,u_s)))
            u_s_dot=np.concatenate([u_s_dot, np.array([u_s_dot_j])])
        
        return u_s_dot


    def H_D(self, q_1, q_B, p_1, p_B, u_B, q_1_0, q_B_0, v_c_u_0, v_c_1_0, c_1, R_0, R_1, v_a, Q_0, beta, v_N, q_1_prev, q_B_prev, v_c_1_prev, v_c_u_prev):
        # TODO: replace as a function of self.K and self.T
        delta = 1
        term_1 = 0.5*((p_B - p_1)/(R_0*delta)) + 0.5*((p_1)**2)/(R_1*delta) 
        term_2 = (c_1/2)*((((q_1 - q_1_prev)/c_1) + v_c_1_prev)**2 - (v_c_1**2))
        term_3 = 0.5*(u_B*(-(q_B - q_B_prev)**2)+2*(-(q_B-q_B_prev)*v_c_u_prev))
        term_4 = (v_N/(beta**2))*(beta*q_B - beta*q_B_prev + (Q_0*beta - Q_0)*np.log((Q_0 - Q_0*beta + beta*q_B)/(Q_0 - Q_0*beta + beta*q_B_prev)))
        term_5 = -(q_B - q_B_prev)*v_a
        H_D = term_1 + term_2 + term_3 + term_4 + term_5
        return H_D
        

