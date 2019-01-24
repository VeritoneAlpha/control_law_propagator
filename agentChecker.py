# Jordan Makansi
# 01/19/19

import numerical_propagator as prp

import unittest   
import numpy as np 
import abc
import scipy as sp
import ode

from blackboard import *

class AgentChecker:
    '''
    This class is used to check the properties of agents:
      - it makes sure the dimensions are consistent for all of the methods
      - it makes sure the outputs of each method are consistent
    ''' 
    def __init__(self, agent): 
        self.agent = agent
        # create fake data for testing inputs
        self.q_s = np.zeros((agent.state_dim))
        self.q_s_dot = np.zeros((agent.state_dim))
        self.p_l = np.zeros((agent.state_dim))
        self.p_mf = np.zeros((agent.state_dim))
        self.u_s = np.zeros((agent.control_dim))

        # need to access the agent's blackboard in order to construct mean field vectors

        
        self.list_of_methods=['L_l', 'L_l_q_dot','L_mf_q_dot','H_l','H_l_nou','H_l_u','q_rhs_H_l_nou',
                 'p_rhs_H_l_nou','q_rhs_H_l_u','p_rhs_H_l_u','q_rhs_H_l','p_rhs_H_l',
                 'qp_rhs_H_l', 'H_mf_nou','H_mf_u','q_rhs_H_mf_u','p_rhs_H_mf_u','q_rhs_H_mf_nou',
                 'p_rhs_H_mf_nou','p_rhs_H_mf','q_rhs_H_mf','qp_rhs_H_mf','qp_rhs','u_rhs']

        # create dictionary of correct inputs, mapping method name to inputs
        self.inputs_dict = {'L_l': (self.q_s, self.q_s_dot, self.u_s),
                 }

        # create list of correct dimensions for outputs
        self.outputs_dict = {'L_l': (0), # remember we check these using np.shape, so if it returns a scalar, then you need to call np.shape(<scalar output>) == ()
                  } 

    def check_All(self):
        '''
        Inputs:
            agent:
        Outputs: 
            method_report: string describing the issues, if any, with each method.  There are four possible outcomes (not mutually exclusive:  No error, missing, incorrect inputs, incorrect outputs.
        ''' 
        #TODO: figure out a clever way to test inputs, rather than just artificially constructing the correct dimensions.  Remember you can use some of the universal ("universal" meaning all agents have them - they are not the same across all agents) attributes of the batteryAgent, such as q_s_0, p_l_0, etc.
        method_report = {}
        existence=True
        outputs=True
        for method in self.list_of_methods:
            # check existence
            if method not in dir(self.agent):
                existence = False
            # get the method
            method_to_call = getattr(self.agent, method)
            # call the method using the fake inputs but with correct dimensions
            inputs = self.inputs_dict[method]        
            result = method_to_call(*inputs)
            
            # check the output dimensions
            outputs_test = self.outputs_dict[method]
            if np.shape(result) != np.shape(outputs_test):
                outputs = False
                
            return_value = (existence, outputs)
            method_report[method]=return_value
        return method_report 
            
