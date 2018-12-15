# Jordan Makansi
# 11/18/18

import numerical_propagator as prp

import unittest   
import numpy as np 
import abc
import scipy as sp
import ode

from blackboard import *


class Agent1:
    
    def __init__(self, blackboard, state_indices, control_indices):
        '''
        state_indices (list of integers): This list tells which states pertain to this agent. e.g. [1,2] would 
        tell us that states 1 and 2 pertain to this agent.
        
        A word on notation:  The notation used for the methods of the agent is:  
            - If it is a partial derivative: <denominator>_rhs_H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g., 
            "qp_rhs_H_l_u" denotes the partial derivative with respect to q and p of the terms in the local Hamiltonian that contain control variables.
            - If it is a hamiltonian: H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g. "H_mf_nou" denotes the mean field hamiltonian with terms not containing u.
        '''
        self.state_indices = state_indices
        self.control_indices = control_indices
        self.bb = blackboard

        # Inputs for numerical propagator
        # qp_vec is going to be [q_s, p_l, p_mf], so it will have dimension = 3*state_dim

        self.q_s_0 = np.array([0])
        self.p_l_0 = np.array([0])
        self.p_mf_0 = np.array([0])
        self.u_s_0 = np.array([0])
        self.qpu_vec = np.hstack([self.q_s_0, self.p_l_0, self.p_mf_0, self.u_s_0])
        self.q_s_dot = np.array([0])  # must have same dimensions as q_s
        self.control_dim = len(self.control_indices)
        self.state_dim = len(self.state_indices)
        self.Gamma = 1
        self.gamma = 1  # function is inputted by the user to compute this.
        self.sync = None # gets filled in when Synchronizer class is initialized
        self.name='Agent1'

        # Inputs for numerical integration
        self.integrateTol = 10**-5
        self.integrateMaxIter = 400

        # Inputs for sliding window
        self.t_0 = 0 
        self.T =  2
        self.K = 4

        self.t_terminal = 4 # terminate entire simulation of this agent
        self.n_s = 10 # number of steps inside of each bucket

        self.validate_dimensions()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
        
    def compute_gamma(self):
        # if only one agent, then gamma = 1
        if len(self.bb.agents) == 1:
            return 1
        q_s, p_l, p_mf, u_s = self.qpu_vec
        q_s_dot = self.q_s_dot
        num = self.L_l(q_s, q_s_dot, u_s)
        denom = 0

        assert len(self.bb.agents) != 0, 'Add agents to your blackboard by calling bb.update_q_p_u_dict(<agent>)'
        for agent in self.bb.agents:
            denom += agent.L_l(q_s, q_s_dot, u_s)
        self.gamma = float(num)/float(denom)
        

    def validate_dimensions(self):
        # TODO: move to parent class "SlidingWindow"
        assert len(self.state_indices) == self.state_dim, 'state dimensions are not consistent.  dimension of state indices is '+str(len(self.state_indices)) +' and state_dim is '+str(self.state_dim)
        assert len(self.control_indices) == len(self.u_s_0), 'control dimensions are not consistent.  dimension of control_indices is '+str(len(self.control_indices)) +' and len(u_0) is '+str(len(self.u_s_0))
        assert len(self.qpu_vec) == 3*self.state_dim + len(self.control_indices), ' control and state dimensions are not consistent with qpu_vec : length of qpu_vec is '+str(len(self.qpu_vec))+ ' and 3*self.state_dim + len(self.control_indices) is ' + str(3*self.state_dim + len(self.control_indices))
    
    '''
    TODO:  
    Add an assertion to check that the dimension of q_s_0, p_mf_0, and u_0:
        - the dimension of state_dim
        - state_indices and control_indices set upon initiation of the Agent
    '''
    
    def L_l(self, q_s, q_s_dot, u_s):
        return 1
    
    def L_l_q_dot(self, q_s, q_s_dot, u_s):
        return q_s
    
    def H_l_nou(self, q_s, p_l, lambda_l):
        return 1

    def p_rhs_H_l_nou(self, q_s, p_l, lambda_l):
        return np.array(p_l)
    
    def q_rhs_H_l_nou(self, q_s, p_l, lambda_l):
        return np.array(p_l)
    
    # There should be one of these defined for each control variable

    def H_l_u_1(self, q_s, p_s):
        return 1
    
    def H_l(self, q_s, p_l, u_s, lambda_l):
        # used in "Construct local Hamiltonian of agent i"
        H_l = self.H_l_nou(q_s, p_l, lambda_l)
        H_l = H_l + self.H_l_u_1(q_s, p_s)*u_s[0]
        return H_l
            
    def compute_lambdas(self, q_s, p_l, u_l):
        # not implemented yet
        return np.ones((1,self.state_dim))
    
    def p_rhs_H_l_u(self, q_s, p_l):
        return np.array(p_l)
    
    def q_rhs_H_l_u(self, q_s, p_l):
        return np.array(p_l)
    
    def H_l_D(self, q_lD, p_lD):
        return np.array(q_lD).dot(p_lD)
        
    def L_l_D(self, q_lD, p_lD):
        # return scalar
        return 1
        
    def L_l_D_q_Dot(self, q_l_D, q_l_D_Dot):
        # return  1-D array of dimension 1 by state_dim,
        # each q_lD is a 1-D array of size 1 by state_dim array
        return 1

    def qp_rhs(self, t, qp_vec, **kwargs):
        # u_s is constant (because of causality, remember?)
        u_s = kwargs['u_0']
        state_dim = kwargs['state_dim']
        q_mf = kwargs['q_mf']
        u_mf = kwargs['u_mf']
        
        # TODO:  get a kwargs working for lambda_l
        lambda_l = 0 # kwargs['lambda_l']
        q_s = qp_vec[:state_dim]
        p_l = qp_vec[state_dim:2*state_dim]
        p_mf = qp_vec[2*state_dim:]
        
        qp_rhs_H_mf = self.qp_rhs_H_mf(q_mf, p_mf, u_mf)
        q_rhs_H_mf = qp_rhs_H_mf[:state_dim]
        p_rhs_H_mf = qp_rhs_H_mf[state_dim:]

        qp_rhs_H_l = self.qp_rhs_H_l(q_s, p_l, u_s, lambda_l)
        q_rhs_H_l = qp_rhs_H_l[:state_dim]
        p_rhs_H_l = qp_rhs_H_l[state_dim:]

        q_s_dot = self.gamma*p_rhs_H_mf + (1-self.gamma)*p_rhs_H_l
        p_mf_dot = q_rhs_H_mf
        p_l_dot = -1*q_rhs_H_l
        
        return np.concatenate([q_s_dot, p_l_dot, p_mf_dot])
    
    def qp_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        #TODO: there is one lambda_l per constraint. need to work out dimensions.
        q_rhs_H_l = self.q_rhs_H_l_nou(q_s, p_l, lambda_l) + sum([self.q_rhs_H_l_u(q_s, p_l)*u_s_i for u_s_i in u_s])
        p_rhs_H_l = self.p_rhs_H_l_nou(q_s, p_l, lambda_l) + sum([self.p_rhs_H_l_u(q_s, p_l)*u_s_i for u_s_i in u_s])
        return np.concatenate([q_rhs_H_l, p_rhs_H_l])


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

    ## Mean Field methods
    def H_MF_nou(self, q_mf, p_mf, u_mf):
        # Shen has u_mf in the flow chart
        return 1

    def H_MF_u(self, q_mf, p_mf, u_mf):
        # q_s, u_mf are vectors for ALL of the states, and controls
        # p_mf is a vector for ONLY the local states/costates
        # length of u_s must match number of terms here
        # some of the elements in u_s are quenched
        return self.H_MF_u_1(q_s, p_mf, u_s)*u_mf[0] + self.H_MF_u_2(q_s, p_mf, u_mf)*u_mf[1]

    def H_MF_u_1(self, q_mf, p_mf):
        return q_mf[0]*q_mf[1]
    
    def H_MF_u_2(self, q_mf, p_mf):
        return q_mf[1]
        
    def qp_rhs_H_mf(self, q_mf, p_mf, u_s):
        # remember that we want to propagate as much as possible together in the same rhs function for numerical purposes
        # remember that q_rhs here is w.r.t p_mf but p_rhs here is w.r.t q_s
        q_H_mf_dot = self.p_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
        p_H_mf_dot = self.q_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
        return np.concatenate([q_H_mf_dot, p_H_mf_dot])
    
    def q_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
        q_rhs_H_mf_u = self.q_rhs_H_mf_u(q_mf, p_mf, u_mf)
        assert np.shape(q_rhs_H_mf_u)==(len(self.control_indices), self.state_dim) # first dimension should be number of controls, inner dimension should be state_dim
        q_rhs_H_mf_u_summed = sum([q_rhs_H_mf_u[i]*u_s[i] for i in range(len(u_s))])
        return self.q_rhs_H_mf_nou(q_mf, p_mf) + q_rhs_H_mf_u_summed
        
    def q_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        # should return something of dimension 
        p_H_mf_u_dot_1 =  p_mf # or something
        return np.array([p_H_mf_u_dot_1])
    
    def p_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
        p_rhs_H_mf_u = self.p_rhs_H_mf_u(q_mf, p_mf, u_mf)
        p_rhs_H_mf_u_summed = sum([p_rhs_H_mf_u[i]*u_s[i] for i in range(len(u_s))])
        return self.p_rhs_H_mf_nou(q_mf, p_mf) + p_rhs_H_mf_u_summed
        
    def p_rhs_H_mf_nou(self, q_mf, p_mf):
        return p_mf # or something

    def q_rhs_H_mf_nou(self, q_mf, p_mf):
        return p_mf

    def p_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        q_H_mf_u_dot = p_mf
        return np.array([q_H_mf_u_dot])

    def L_mf_q_dot(self, q_mf, q_mf_dot, u_mf):
        # q_mf_dot, q_mf (inputs) here will be vectors with ALL of the states
        # u_mf is a vector of ALL of the controls
        # extract q_s from q_mf
        
        # note that these methods must return vectors that are of local dimension - state_dim - even though they take in vectors of dimension for all the states
        # the user needs to be aware of the indices the correspond to each state
        
        def L_l_q_dot_agent_1(q_mf, q_mf_dot, u_mf):
            return np.array(q_mf[0])
        
        def L_l_q_dot_agent_2(q_mf, q_mf_dot, u_mf):
            return np.array(q_mf[1])
        
        L_mf_total_q_dot = np.zeros(self.state_dim)

        # agent 1
        L_mf_total_q_dot += L_l_q_dot_agent_1(q_mf, q_mf_dot, u_mf)

        # agent 2
        L_mf_total_q_dot += L_l_q_dot_agent_2(q_mf, q_mf_dot, u_mf)
        assert np.shape(L_mf_total_q_dot)[0] == self.state_dim, 'dimensions of L_mf_total_q_dot must match those of the local state, currently the dimensions are ' +str(np.shape(L_mf_total_q_dot)[0])
        return L_mf_total_q_dot


class Agent2(object):
    
    def __init__(self, blackboard, state_indices, control_indices):
        '''
        state_indices (list of integers): This list tells which states pertain to this agent. e.g. [1,2] would 
        tell us that states 1 and 2 pertain to this agent.
        
        A word oNn notation:  The notation used for the methods of the agent is:  
            - If it is a partial derivative: <denominator>_rhs_H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g., 
            "qp_rhs_H_l_u" denotes the partial derivative with respect to q and p of the terms in the local Hamiltonian that contain control variables.
            - If it is a hamiltonian: H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g. "H_mf_nou" denotes the mean field hamiltonian
            with terms not containing u.
        '''
        self.state_indices = state_indices
        self.control_indices = control_indices
        self.bb = blackboard

        # Inputs for numerical propagator
        # qp_vec is going to be [q_s, p_l, p_mf], so it will have dimension = 3*state_dim

        self.q_s_0 = np.array([0,2])
        self.p_l_0 = np.array([0,3])
        self.p_mf_0 = np.array([0,1])
        self.u_s_0 = np.array([0])
        self.qpu_vec = np.hstack([self.q_s_0, self.p_l_0, self.p_mf_0, self.u_s_0])
        self.state_dim = len(self.state_indices)
        self.control_dim = len(self.control_indices)
        self.Gamma = 1 
        self.gamma = 1 # gets computed each time the agent is visited
        self.q_s_dot = np.array([0,1])  # must have same dimensions as q_s
        self.sync = None # gets Synchronizer class is initialized
        self.name='Agent2'
        
        # Inputs for numerical integration
        self.integrateTol = 10**-5
        self.integrateMaxIter = 400

        # Inputs for sliding window
        self.t_0 = 0
        self.T = 2
        self.K = 4

        self.t_terminal = 2
        self.n_s = 10

        self.validate_dimensions() 

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def validate_dimensions(self):
        # TODO: move to parent class "SlidingWindow"
        assert len(self.state_indices) == self.state_dim, 'state dimensions are not consistent.  dimension of state indices is '+str(len(self.state_indices)) +' and state_dim is '+str(self.state_dim)
        assert len(self.control_indices) == len(self.u_s_0), 'control dimensions are not consistent.  dimension of control_indices is '+str(len(self.control_indices)) +' and len(u_0) is '+str(len(self.u_s_0))
        assert len(self.qpu_vec) == 3*self.state_dim + len(self.control_indices), ' control and state dimensions are not consistent with qpu_vec : length of qpu_vec is '+str(len(self.qpu_vec))+ ' and 3*self.state_dim + len(self.control_indices) is ' + str(3*self.state_dim + len(self.control_indices))
    
    '''
    TODO:  
    Add an assertion to check that the dimension of q_s_0, p_mf_0, and u_0:
        - the dimension of state_dim    
        - state_indices and control_indices set upon initiation of the Agent
    '''
    
    def L_l(self, q_s, q_s_dot, u_s):
        return 1
    
    def L_l_q_dot(self, q_s, q_s_dot, u_s):
        return q_s
    
    def H_l_nou(self, q_s, p_l, lambda_l):
        return 1

    def H_l_u(self, q_s, p_l):
        return np.array([0])

    def q_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        q_rhs_H_l_u = self.q_rhs_H_l_u(q_s, p_l)
        q_rhs_H_l_u_summed = sum([q_rhs_H_l_u[i]*u_s[i] for i in range(len(u_s))])
        return self.q_rhs_H_l_nou(q_s, p_l, lambda_l) + q_rhs_H_l_u_summed

    def qp_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        # TODO: there is one lambda_l per constraint. need to work out dimensions.
        q_H_l_dot = self.p_rhs_H_l(q_s, p_l, u_s, lambda_l)
        p_H_l_dot = self.q_rhs_H_l(q_s, p_l, u_s, lambda_l)
        return np.concatenate([q_H_l_dot, p_H_l_dot])

    def q_rhs_H_l_u(self, q_s, p_l):
        # this must be 2D array, so wrap it in np.array
        q_rhs_H_l_u = np.array([q_s])
        return q_rhs_H_l_u
    
    def q_rhs_H_l_nou(self, q_s, p_l, lambda_l):
        # this needs to be 1d array so no need to wrap it in np.array, unlike q_rhs_H_l_u
        q_rhs_H_l_nou = q_s
        return q_rhs_H_l_nou
    
    def p_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        p_rhs_H_l_u = self.p_rhs_H_l_u(q_s, p_l)
        p_rhs_H_l_u_summed = sum([p_rhs_H_l_u[i]*u_s[i] for i in range(len(u_s))])
        return self.p_rhs_H_l_nou(q_s, p_l, lambda_l) + p_rhs_H_l_u_summed

    def p_rhs_H_l_u(self, q_s, p_l):
        # this must be 2D array, so wrap it in np.array
        p_rhs_H_l_u = np.array([q_s])
        return p_rhs_H_l_u
    
    def p_rhs_H_l_nou(self, q_s, p_l, lambda_l):
        # this needs to be 1d array so no need to wrap it in np.array, unlike q_rhs_H_l_u
        p_rhs_H_l_nou = q_s
        return p_rhs_H_l_nou

    def H_l(self, q_s, p_l, lambda_l, u_s):
        # used in "Construct local Hamiltonian of agent i"
        H_l_nou = self.H_l_nou(q_s, p_l, lambda_l)
        H_l_u = self.H_l_u(q_s, p_l, lambda_l)
        H_l_u_summed = sum([H_l_u[i]*u_s[i] for i in range(len(u_s))])
        H_l = H_l_nou + H_l_u_summed
        return H_l            
    
    def compute_lambdas(self, q_s, p_l, u_l):
        # not implemented yet
        return np.ones((1,self.state_dim))
    
    def H_l_D(self, q_lD, p_lD):
        return np.array(q_lD).dot(p_lD)
        
    def L_l_D(self, q_lD, p_lD):
        # return scalar
        return 1
        
    def L_l_D_q_Dot(self, q_lD, p_lD):
        # return 1 by state_dim, 1-D array
        # each q_lD is a 1-D array of size 1 by state_dim array
        return 1
    
    def qp_rhs(self, t, qp_vec, **kwargs):
        # u_s is constant (because of causality, remember?)
        u_s = kwargs['u_0']
        state_dim = kwargs['state_dim']
        q_mf = kwargs['q_mf']
        u_mf = kwargs['u_mf']
        
        # TODO:  get a kwargs working for lambda_l
        lambda_l = 0 # kwargs['lambda_l']
        q_s = qp_vec[:state_dim]
        p_l = qp_vec[state_dim:2*state_dim]
        p_mf = qp_vec[2*state_dim:]

        qp_rhs_H_mf = self.qp_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
        q_rhs_H_mf = qp_rhs_H_mf[:state_dim]
        p_rhs_H_mf = qp_rhs_H_mf[state_dim:]
        
        qp_rhs_H_l = self.qp_rhs_H_l(q_s, p_l, u_s, lambda_l)
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


    ## Mean Field methods
    def H_MF_nou(self, q_mf, p_mf, u_mf):
        return 1

    def H_MF_u(self, q_mf, p_mf):
        # q_mf, u_mf are vectors for ALL of the states, and controls
        # retrns a numpy array with each element corresponding to H_mf for a particular control variable, j
        return np.array([self.H_MF_u_1(q_mf, p_mf)]) #*u_mf[0]])  # + self.H_MF_u_2(q_mf, p_mf)*u_mf[1]

    def H_MF_u_1(self, q_mf, p_mf):
        return q_mf[0]*q_mf[1]
    
#     def H_MF_u_2(self, q_mf, p_mf):
#         return q_mf[1]
        
    def qp_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        # remember that we want to propagate as much as possible together in the same rhs function for numerical purposes
        # remember that q_rhs here is w.r.t p_mf but p_rhs here is w.r.t q_s
        q_H_mf_dot = self.p_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
        p_H_mf_dot = self.q_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
        return np.concatenate([q_H_mf_dot, p_H_mf_dot])
    
    def q_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
        q_rhs_H_mf_u = self.q_rhs_H_mf_u(q_mf, p_mf, u_mf)
        assert np.shape(q_rhs_H_mf_u)==(len(self.control_indices), self.state_dim) # first dimension should be number of controls, inner dimension should be state_dim
        q_rhs_H_mf_u_summed = sum([q_rhs_H_mf_u[i]*u_s[i] for i in range(len(u_s))])
        return self.q_rhs_H_mf_nou(q_mf, p_mf) + q_rhs_H_mf_u_summed
        
    def q_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        # this method is will return a concatenation of all of the partial derivatives for each of the controls
        # each of the partial derivatives is of dimension state_dim
        # this means that this method will return a 2D array:
        #    - first dimension is the index of the control
        #    - second dimension is the index of the state
        p_H_mf_u_dot_1 = p_mf # must be of dimension state_dim
        q_rhs_H_mf_u = np.array([p_H_mf_u_dot_1])
        return q_rhs_H_mf_u

    def p_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
        p_rhs_H_mf_u = self.p_rhs_H_mf_u(q_mf, p_mf, u_mf)
        p_rhs_H_mf_u_summed = sum([p_rhs_H_mf_u[i]*u_s[i] for i in range(len(u_s))])
        return self.p_rhs_H_mf_nou(q_mf, p_mf) + p_rhs_H_mf_u_summed
        
    def p_rhs_H_mf_nou(self, q_mf, p_mf):
        return  p_mf # or something

    def q_rhs_H_mf_nou(self, q_mf, p_mf):
        return  p_mf

    def p_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        q_H_mf_u_dot = p_mf
        return np.array([q_H_mf_u_dot])

    def L_mf_q_dot(self, q_mf, q_mf_dot, u_mf):
        # q_mf_dot, q_mf (inputs) here will be vectors with ALL of the states
        # u_mf is a vector of ALL of the controls
        # extract q_s from q_mf
        
        # note that these methods must return vectors that are of local dimension - state_dim - even though they take in vectors of dimension for all the states
        # the user needs to be aware of the indices the correspond to each state

        def L_l_q_dot_agent_1(q_mf, q_mf_dot, u_mf):
            return np.concatenate([np.array([q_mf[0]]),np.array([q_mf[1]])])
        
        def L_l_q_dot_agent_2(q_mf, q_mf_dot, u_mf):
            return np.concatenate([np.array([q_mf[0]]),np.array([q_mf[1]])])
        
        L_mf_total_q_dot = np.zeros(self.state_dim)

        # agent 1
        L_mf_total_q_dot += L_l_q_dot_agent_1(q_mf, q_mf_dot, u_mf)

        # agent 2
        L_mf_total_q_dot += L_l_q_dot_agent_2(q_mf, q_mf_dot, u_mf)

        assert np.shape(L_mf_total_q_dot)[0] == self.state_dim, 'dimensions of L_mf_total_q_dot must match those of the local state, currently the dimensions are ' +str(np.shape(L_mf_total_q_dot)[0])
        return L_mf_total_q_dot


class Agent3:
    
    def __init__(self, blackboard, state_indices, control_indices):
        '''
        state_indices (list of integers): This list tells which states pertain to this agent. e.g. [1,2] would 
        tell us that states 1 and 2 pertain to this agent.
        
        A word oNn notation:  The notation used for the methods of the agent is:  
            - If it is a partial derivative: <denominator>_rhs_H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g., 
            "qp_rhs_H_l_u" denotes the partial derivative with respect to q and p of the terms in the local Hamiltonian that contain control variables.
            - If it is a hamiltonian: H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g. "H_mf_nou" denotes the mean field hamiltonian
            with terms not containing u.
        '''
        self.state_indices = state_indices
        self.control_indices = control_indices
        self.bb = blackboard

        # Inputs for numerical propagator
        # qp_vec is going to be [q_s, p_l, p_mf], so it will have dimension = 3*state_dim

        self.q_s_0 = np.array([0,2])
        self.p_l_0 = np.array([0,3])
        self.p_mf_0 = np.array([0,1])
        self.u_s_0 = np.array([0, 0])
        self.qpu_vec = np.hstack([self.q_s_0, self.p_l_0, self.p_mf_0, self.u_s_0])
        self.state_dim = len(self.state_indices)
        self.control_dim = len(self.control_indices)
        self.Gamma = 1 
        self.gamma = 1 # gets computed each time the agent is visited
        self.q_s_dot = np.array([0,1])  # must have same dimensions as q_s
        self.sync = None # gets Synchronizer class is initialized
        self.name='Agent3'
        
        # Inputs for numerical integration
        self.integrateTol = 10**-5
        self.integrateMaxIter = 400

        # Inputs for sliding window
        self.t_0 = 0
        self.T = 2
        self.K = 4

        self.t_terminal = 2
        self.n_s = 10

        self.validate_dimensions() 

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def validate_dimensions(self):
        # TODO: move to parent class "SlidingWindow"
        assert len(self.state_indices) == self.state_dim, 'state dimensions are not consistent.  dimension of state indices is '+str(len(self.state_indices)) +' and state_dim is '+str(self.state_dim)
        assert len(self.control_indices) == len(self.u_s_0), 'control dimensions are not consistent.  dimension of control_indices is '+str(len(self.control_indices)) +' and len(u_0) is '+str(len(self.u_s_0))
        assert len(self.qpu_vec) == 3*self.state_dim + len(self.control_indices), ' control and state dimensions are not consistent with qpu_vec : length of qpu_vec is '+str(len(self.qpu_vec))+ ' and 3*self.state_dim + len(self.control_indices) is ' + str(3*self.state_dim + len(self.control_indices))
    
    '''
    TODO:  
    Add an assertion to check that the dimension of q_s_0, p_mf_0, and u_0:
        - the dimension of state_dim    
        - state_indices and control_indices set upon initiation of the Agent
    '''
    
    def L_l(self, q_s, q_s_dot, u_s):
        return 1
    
    def L_l_q_dot(self, q_s, q_s_dot, u_s):
        return q_s
    
    def H_l_nou(self, q_s, p_l, lambda_l):
        return 1

    def H_l_u(self, q_s, p_l):
        return np.array([0, 0])

    def q_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        q_rhs_H_l_u = self.q_rhs_H_l_u(q_s, p_l)
        q_rhs_H_l_u_summed = sum([q_rhs_H_l_u[i]*u_s[i] for i in range(len(u_s))])
        return self.q_rhs_H_l_nou(q_s, p_l, lambda_l) + q_rhs_H_l_u_summed

    def qp_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        # TODO: there is one lambda_l per constraint. need to work out dimensions.
        q_H_l_dot = self.p_rhs_H_l(q_s, p_l, u_s, lambda_l)
        p_H_l_dot = self.q_rhs_H_l(q_s, p_l, u_s, lambda_l)
        return np.concatenate([q_H_l_dot, p_H_l_dot])

    def q_rhs_H_l_u(self, q_s, p_l):
        # this must be 2D array, so wrap it in np.array
        q_rhs_H_l_u = np.array([q_s, p_l])
        return q_rhs_H_l_u
    
    def q_rhs_H_l_nou(self, q_s, p_l, lambda_l):
        # this needs to be 1d array so no need to wrap it in np.array, unlike q_rhs_H_l_u
        q_rhs_H_l_nou = q_s
        return q_rhs_H_l_nou
    
    def p_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        p_rhs_H_l_u = self.p_rhs_H_l_u(q_s, p_l)
        p_rhs_H_l_u_summed = sum([p_rhs_H_l_u[i]*u_s[i] for i in range(len(u_s))])
        return self.p_rhs_H_l_nou(q_s, p_l, lambda_l) + p_rhs_H_l_u_summed

    def p_rhs_H_l_u(self, q_s, p_l):
        # this must be 2D array, so wrap it in np.array
        p_rhs_H_l_u = np.array([q_s, q_s])
        return p_rhs_H_l_u
    
    def p_rhs_H_l_nou(self, q_s, p_l, lambda_l):
        # this needs to be 1d array so no need to wrap it in np.array, unlike q_rhs_H_l_u
        p_rhs_H_l_nou = q_s
        return p_rhs_H_l_nou

    def H_l(self, q_s, p_l, lambda_l, u_s):
        # used in "Construct local Hamiltonian of agent i"
        H_l_nou = self.H_l_nou(q_s, p_l, lambda_l)
        H_l_u = self.H_l_u(q_s, p_l, lambda_l)
        H_l_u_summed = sum([H_l_u[i]*u_s[i] for i in range(len(u_s))])
        H_l = H_l_nou + H_l_u_summed
        return H_l            
    
    def compute_lambdas(self, q_s, p_l, u_l):
        # not implemented yet
        return np.ones((1,self.state_dim))
    
    def H_l_D(self, q_lD, p_lD):
        return np.array(q_lD).dot(p_lD)
        
    def L_l_D(self, q_lD, p_lD):
        # return scalar
        return 1
        
    def L_l_D_q_Dot(self, q_lD, p_lD):
        # return 1 by state_dim, 1-D array
        # each q_lD is a 1-D array of size 1 by state_dim array
        return 1
    
    def qp_rhs(self, t, qp_vec, **kwargs):
        # u_s is constant (because of causality, remember?)
        u_s = kwargs['u_0']
        state_dim = kwargs['state_dim']
        q_mf = kwargs['q_mf']
        u_mf = kwargs['u_mf']
        
        # TODO:  get a kwargs working for lambda_l
        lambda_l = 0 # kwargs['lambda_l']
        q_s = qp_vec[:state_dim]
        p_l = qp_vec[state_dim:2*state_dim]
        p_mf = qp_vec[2*state_dim:]

        qp_rhs_H_mf = self.qp_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
        q_rhs_H_mf = qp_rhs_H_mf[:state_dim]
        p_rhs_H_mf = qp_rhs_H_mf[state_dim:]
        
        qp_rhs_H_l = self.qp_rhs_H_l(q_s, p_l, u_s, lambda_l)
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


    ## Mean Field methods
    def H_MF_nou(self, q_mf, p_mf, u_mf):
        return 1

    def H_MF_u(self, q_mf, p_mf):
        # q_mf, u_mf are vectors for ALL of the states, and controls
        # retrns a numpy array with each element corresponding to H_mf for a particular control variable, j
        return np.array([self.H_MF_u_1(q_mf, p_mf), self.H_MF_u_2(q_mf, p_mf)]) 

    def H_MF_u_1(self, q_mf, p_mf):
        return q_mf[0]*q_mf[1]
    
    def H_MF_u_2(self, q_mf, p_mf):
        return q_mf[1]
        
    def qp_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        # remember that we want to propagate as much as possible together in the same rhs function for numerical purposes
        # remember that q_rhs here is w.r.t p_mf but p_rhs here is w.r.t q_s
        q_H_mf_dot = self.p_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
        p_H_mf_dot = self.q_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
        return np.concatenate([q_H_mf_dot, p_H_mf_dot])
    
    def q_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
        q_rhs_H_mf_u = self.q_rhs_H_mf_u(q_mf, p_mf, u_mf)
        assert np.shape(q_rhs_H_mf_u)==(len(self.control_indices), self.state_dim) # first dimension should be number of controls, inner dimension should be state_dim
        q_rhs_H_mf_u_summed = sum([q_rhs_H_mf_u[i]*u_s[i] for i in range(len(u_s))])
        return self.q_rhs_H_mf_nou(q_mf, p_mf) + q_rhs_H_mf_u_summed
        
    def q_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        # this method is will return a concatenation of all of the partial derivatives for each of the controls
        # each of the partial derivatives is of dimension state_dim
        # this means that this method will return a 2D array:
        #    - first dimension is the index of the control
        #    - second dimension is the index of the state
        p_H_mf_u_dot_1 = p_mf # must be of dimension state_dim
        q_rhs_H_mf_u = np.array([p_H_mf_u_dot_1, p_H_mf_u_dot_1])
        return q_rhs_H_mf_u

    def p_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
        p_rhs_H_mf_u = self.p_rhs_H_mf_u(q_mf, p_mf, u_mf)
        p_rhs_H_mf_u_summed = sum([p_rhs_H_mf_u[i]*u_s[i] for i in range(len(u_s))])
        return self.p_rhs_H_mf_nou(q_mf, p_mf) + p_rhs_H_mf_u_summed
        
    def p_rhs_H_mf_nou(self, q_mf, p_mf):
        return  p_mf # or something

    def q_rhs_H_mf_nou(self, q_mf, p_mf):
        return  p_mf

    def p_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        q_H_mf_u_dot = p_mf
        return np.array([q_H_mf_u_dot, q_H_mf_u_dot])

    def L_mf_q_dot(self, q_mf, q_mf_dot, u_mf):
        # q_mf_dot, q_mf (inputs) here will be vectors with ALL of the states
        # u_mf is a vector of ALL of the controls
        # extract q_s from q_mf
        
        # note that these methods must return vectors that are of local dimension - state_dim - even though they take in vectors of dimension for all the states
        # the user needs to be aware of the indices the correspond to each state

        def L_l_q_dot_agent_1(q_mf, q_mf_dot, u_mf):
            return np.concatenate([np.array([q_mf[0]]),np.array([q_mf[1]])])
        
        def L_l_q_dot_agent_2(q_mf, q_mf_dot, u_mf):
            return np.concatenate([np.array([q_mf[0]]),np.array([q_mf[1]])])
        
        L_mf_total_q_dot = np.zeros(self.state_dim)

        # agent 1
        L_mf_total_q_dot += L_l_q_dot_agent_1(q_mf, q_mf_dot, u_mf)

        # agent 2
        L_mf_total_q_dot += L_l_q_dot_agent_2(q_mf, q_mf_dot, u_mf)

        assert np.shape(L_mf_total_q_dot)[0] == self.state_dim, 'dimensions of L_mf_total_q_dot must match those of the local state, currently the dimensions are ' +str(np.shape(L_mf_total_q_dot)[0])
        return L_mf_total_q_dot

class Agent4(Agent2):
    
    def __init__(self, blackboard, state_indices, control_indices):
#        '''
#        state_indices (list of integers): This list tells which states pertain to this agent. e.g. [1,2] would 
#        tell us that states 1 and 2 pertain to this agent.
#        
#        A word oNn notation:  The notation used for the methods of the agent is:  
#            - If it is a partial derivative: <denominator>_rhs_H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g., 
#            "qp_rhs_H_l_u" denotes the partial derivative with respect to q and p of the terms in the local Hamiltonian that contain control variables.
#            - If it is a hamiltonian: H_<type of hamiltonian (l, mf, or s)>_<nou or u>.  e.g. "H_mf_nou" denotes the mean field hamiltonian
#            with terms not containing u.
#        '''
#        self.state_indices = state_indices
#        self.control_indices = control_indices
#        self.bb = blackboard
#
#        # Inputs for numerical propagator
#        # qp_vec is going to be [q_s, p_l, p_mf], so it will have dimension = 3*state_dim
#
#        self.q_s_0 = np.array([0,2])
#        self.p_l_0 = np.array([0,3])
#        self.p_mf_0 = np.array([0,1])
         self.u_s_0 = np.array([0, 0])
#        self.qpu_vec = np.hstack([self.q_s_0, self.p_l_0, self.p_mf_0, self.u_s_0])
#        self.state_dim = len(self.state_indices)
#        self.control_dim = len(self.control_indices)
#        self.Gamma = 1 
#        self.gamma = 1 # gets computed each time the agent is visited
#        self.q_s_dot = np.array([0,1])  # must have same dimensions as q_s
#        self.sync = None # gets Synchronizer class is initialized
         self.name='Agent4'
#        
#        # Inputs for numerical integration
#        self.integrateTol = 10**-5
#        self.integrateMaxIter = 400
#
#        # Inputs for sliding window
#        self.t_0 = 0
#        self.T = 2
#        self.K = 4
#
#        self.t_terminal = 2
#        self.n_s = 10
         super(Agent4,self).__init__(blackboard, state_indices, control_indices)
         import pdb; pdb.set_trace()
         self.validate_dimensions() 

#    def __repr__(self):
#        return self.name
#
#    def __str__(self):
#        return self.name
#
#    def validate_dimensions(self):
#        # TODO: move to parent class "SlidingWindow"
#        assert len(self.state_indices) == self.state_dim, 'state dimensions are not consistent.  dimension of state indices is '+str(len(self.state_indices)) +' and state_dim is '+str(self.state_dim)
#        assert len(self.control_indices) == len(self.u_s_0), 'control dimensions are not consistent.  dimension of control_indices is '+str(len(self.control_indices)) +' and len(u_0) is '+str(len(self.u_s_0))
#        assert len(self.qpu_vec) == 3*self.state_dim + len(self.control_indices), ' control and state dimensions are not consistent with qpu_vec : length of qpu_vec is '+str(len(self.qpu_vec))+ ' and 3*self.state_dim + len(self.control_indices) is ' + str(3*self.state_dim + len(self.control_indices))
#    
#    '''
#    TODO:  
#    Add an assertion to check that the dimension of q_s_0, p_mf_0, and u_0:
#        - the dimension of state_dim    
#        - state_indices and control_indices set upon initiation of the Agent
#    '''
#    
#    def L_l(self, q_s, q_s_dot, u_s):
#        return 1
#    
#    def L_l_q_dot(self, q_s, q_s_dot, u_s):
#        return q_s
#    
#    def H_l_nou(self, q_s, p_l, lambda_l):
#        return 1
#
    def H_l_u(self, q_s, p_l):
        return np.array([0, 0])

#    def q_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
#        q_rhs_H_l_u = self.q_rhs_H_l_u(q_s, p_l)
#        q_rhs_H_l_u_summed = sum([q_rhs_H_l_u[i]*u_s[i] for i in range(len(u_s))])
#        return self.q_rhs_H_l_nou(q_s, p_l, lambda_l) + q_rhs_H_l_u_summed
#
#    def qp_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
#        # TODO: there is one lambda_l per constraint. need to work out dimensions.
#        q_H_l_dot = self.p_rhs_H_l(q_s, p_l, u_s, lambda_l)
#        p_H_l_dot = self.q_rhs_H_l(q_s, p_l, u_s, lambda_l)
#        return np.concatenate([q_H_l_dot, p_H_l_dot])

    def q_rhs_H_l_u(self, q_s, p_l):
        # this must be 2D array, so wrap it in np.array
        q_rhs_H_l_u = np.array([q_s, p_l])
        return q_rhs_H_l_u
    
#    def q_rhs_H_l_nou(self, q_s, p_l, lambda_l):
#        # this needs to be 1d array so no need to wrap it in np.array, unlike q_rhs_H_l_u
#        q_rhs_H_l_nou = q_s
#        return q_rhs_H_l_nou
#    
#    def p_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
#        p_rhs_H_l_u = self.p_rhs_H_l_u(q_s, p_l)
#        p_rhs_H_l_u_summed = sum([p_rhs_H_l_u[i]*u_s[i] for i in range(len(u_s))])
#        return self.p_rhs_H_l_nou(q_s, p_l, lambda_l) + p_rhs_H_l_u_summed
#
    def p_rhs_H_l_u(self, q_s, p_l):
        # this must be 2D array, so wrap it in np.array
        p_rhs_H_l_u = np.array([q_s, q_s])
        return p_rhs_H_l_u
    
#    def p_rhs_H_l_nou(self, q_s, p_l, lambda_l):
#        # this needs to be 1d array so no need to wrap it in np.array, unlike q_rhs_H_l_u
#        p_rhs_H_l_nou = q_s
#        return p_rhs_H_l_nou
#
#    def H_l(self, q_s, p_l, lambda_l, u_s):
#        # used in "Construct local Hamiltonian of agent i"
#        H_l_nou = self.H_l_nou(q_s, p_l, lambda_l)
#        H_l_u = self.H_l_u(q_s, p_l, lambda_l)
#        H_l_u_summed = sum([H_l_u[i]*u_s[i] for i in range(len(u_s))])
#        H_l = H_l_nou + H_l_u_summed
#        return H_l            
#    
#    def compute_lambdas(self, q_s, p_l, u_l):
#        # not implemented yet
#        return np.ones((1,self.state_dim))
#    
#    def H_l_D(self, q_lD, p_lD):
#        return np.array(q_lD).dot(p_lD)
#        
#    def L_l_D(self, q_lD, p_lD):
#        # return scalar
#        return 1
#        
#    def L_l_D_q_Dot(self, q_lD, p_lD):
#        # return 1 by state_dim, 1-D array
#        # each q_lD is a 1-D array of size 1 by state_dim array
#        return 1
#    
#    def qp_rhs(self, t, qp_vec, **kwargs):
#        # u_s is constant (because of causality, remember?)
#        u_s = kwargs['u_0']
#        state_dim = kwargs['state_dim']
#        q_mf = kwargs['q_mf']
#        u_mf = kwargs['u_mf']
#        
#        # TODO:  get a kwargs working for lambda_l
#        lambda_l = 0 # kwargs['lambda_l']
#        q_s = qp_vec[:state_dim]
#        p_l = qp_vec[state_dim:2*state_dim]
#        p_mf = qp_vec[2*state_dim:]
#
#        qp_rhs_H_mf = self.qp_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
#        q_rhs_H_mf = qp_rhs_H_mf[:state_dim]
#        p_rhs_H_mf = qp_rhs_H_mf[state_dim:]
#        
#        qp_rhs_H_l = self.qp_rhs_H_l(q_s, p_l, u_s, lambda_l)
#        q_rhs_H_l = qp_rhs_H_l[:state_dim]
#        p_rhs_H_l = qp_rhs_H_l[state_dim:]
#
#        q_s_dot = self.gamma*p_rhs_H_mf + (1-self.gamma)*p_rhs_H_l
#        p_mf_dot = q_rhs_H_mf
#        p_l_dot = -1*q_rhs_H_l
#
#        return np.concatenate([q_s_dot, p_l_dot, p_mf_dot])
#    
#
#    def u_rhs(self, t, u_vec, **kwargs):
#        u_s = kwargs['u_0']
#        state_dim = kwargs['state_dim']
#        q_mf_dot = kwargs['q_mf_dot']
#        q_s_dot = kwargs['q_s_dot']
#        p_l_dot = kwargs['p_l_dot']
#        p_mf_dot = kwargs['p_mf_dot']
#        q_mf = kwargs['q_mf']
#        u_mf = kwargs['u_mf']
#        qp_vec = kwargs['qp_vec']
#        H_l_D = kwargs['H_l_D']
#        Beta_mf = kwargs['Beta_mf']
#        Beta_l = kwargs['Beta_l']
#        alpha_mf = kwargs['alpha_mf']
#        alpha_l = kwargs['alpha_l']
#        q_s = qp_vec[:state_dim]
#        p_l = qp_vec[state_dim:2*state_dim]
#        p_mf = qp_vec[2*state_dim:]
#        u_s_dot = np.array([])
#        for j in range(self.control_dim):
#            '''for each control, we need to:
#                1) Compute and get a 1D np.array for each of alpha_l_j, etc.
#                2) Compute u_s_dot_j = -1*self.Gamma*(self.gamma*(alpha_mf_j + np.dot(Beta_mf_j, u_s)) + (1-self.gamma)*(alpha_l_j + np.dot(Beta_l_j,u_s)))
#                3) Concatenate all of the u_s_dot_j to construct u_s_dot in a 1D np.array
#            '''
#            
#            Beta_mf_j,Beta_l_j = Beta_mf[j], Beta_l[j] 
#            alpha_mf_j, alpha_l_j = alpha_mf[j], alpha_l[j]
#            # Beta_mf_j, Beta_l_j should be vectors
#            # alpha_mf_j, alpha_l_j should be scalars
#            u_s_dot_j = -1*self.Gamma*(self.gamma*(alpha_mf_j + np.dot(Beta_mf_j, u_s)) + (1-self.gamma)*(alpha_l_j + np.dot(Beta_l_j,u_s)))
#            u_s_dot=np.concatenate([u_s_dot, np.array([u_s_dot_j])])
#        
#        return u_s_dot
#
#
#    ## Mean Field methods
#    def H_MF_nou(self, q_mf, p_mf, u_mf):
#        return 1

    def H_MF_u(self, q_mf, p_mf):
        # q_mf, u_mf are vectors for ALL of the states, and controls
        # retrns a numpy array with each element corresponding to H_mf for a particular control variable, j
        return np.array([self.H_MF_u_1(q_mf, p_mf), self.H_MF_u_2(q_mf, p_mf)]) 

    def H_MF_u_1(self, q_mf, p_mf):
        return q_mf[0]*q_mf[1]
    
    def H_MF_u_2(self, q_mf, p_mf):
        return q_mf[1]
        
#    def qp_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
#        # remember that we want to propagate as much as possible together in the same rhs function for numerical purposes
#        # remember that q_rhs here is w.r.t p_mf but p_rhs here is w.r.t q_s
#        q_H_mf_dot = self.p_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
#        p_H_mf_dot = self.q_rhs_H_mf(q_mf, p_mf, u_mf, u_s)
#        return np.concatenate([q_H_mf_dot, p_H_mf_dot])
#    
#    def q_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
#        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
#        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
#        q_rhs_H_mf_u = self.q_rhs_H_mf_u(q_mf, p_mf, u_mf)
#        assert np.shape(q_rhs_H_mf_u)==(len(self.control_indices), self.state_dim) # first dimension should be number of controls, inner dimension should be state_dim
#        q_rhs_H_mf_u_summed = sum([q_rhs_H_mf_u[i]*u_s[i] for i in range(len(u_s))])
#        return self.q_rhs_H_mf_nou(q_mf, p_mf) + q_rhs_H_mf_u_summed
        
    def q_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        # this method is will return a concatenation of all of the partial derivatives for each of the controls
        # each of the partial derivatives is of dimension state_dim
        # this means that this method will return a 2D array:
        #    - first dimension is the index of the control
        #    - second dimension is the index of the state
        p_H_mf_u_dot_1 = p_mf # must be of dimension state_dim
        q_rhs_H_mf_u = np.array([p_H_mf_u_dot_1, p_H_mf_u_dot_1])
        return q_rhs_H_mf_u
#
#    def p_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
#        # q_rhs_H_mf is the derivative wrt each of the local variables, so it will return something of dimension state_dim
#        # q_rhs_H_mf_u returns the partial derivatives wrt each control, concatenated together
#        p_rhs_H_mf_u = self.p_rhs_H_mf_u(q_mf, p_mf, u_mf)
#        p_rhs_H_mf_u_summed = sum([p_rhs_H_mf_u[i]*u_s[i] for i in range(len(u_s))])
#        return self.p_rhs_H_mf_nou(q_mf, p_mf) + p_rhs_H_mf_u_summed
#        
#    def p_rhs_H_mf_nou(self, q_mf, p_mf):
#        return  p_mf # or something
#
#    def q_rhs_H_mf_nou(self, q_mf, p_mf):
#        return  p_mf

    def p_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        q_H_mf_u_dot = p_mf
        return np.array([q_H_mf_u_dot, q_H_mf_u_dot])

#    def L_mf_q_dot(self, q_mf, q_mf_dot, u_mf):
#        # q_mf_dot, q_mf (inputs) here will be vectors with ALL of the states
#        # u_mf is a vector of ALL of the controls
#        # extract q_s from q_mf
#        # note that these methods must return vectors that are of local dimension - state_dim - even though they take in vectors of dimension for all the states
#        # the user needs to be aware of the indices the correspond to each state
#
#        def L_l_q_dot_agent_1(q_mf, q_mf_dot, u_mf):
#            return np.concatenate([np.array([q_mf[0]]),np.array([q_mf[1]])])
#        
#        def L_l_q_dot_agent_2(q_mf, q_mf_dot, u_mf):
#            return np.concatenate([np.array([q_mf[0]]),np.array([q_mf[1]])])
#        
#        L_mf_total_q_dot = np.zeros(self.state_dim)
#
#        # agent 1
#        L_mf_total_q_dot += L_l_q_dot_agent_1(q_mf, q_mf_dot, u_mf)
#
#        # agent 2
#        L_mf_total_q_dot += L_l_q_dot_agent_2(q_mf, q_mf_dot, u_mf)
#
#        assert np.shape(L_mf_total_q_dot)[0] == self.state_dim, 'dimensions of L_mf_total_q_dot must match those of the local state, currently the dimensions are ' +str(np.shape(L_mf_total_q_dot)[0])
#        return L_mf_total_q_dot
