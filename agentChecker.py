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
        self.p_l = np.zeros((agent.state_dim))
        self.p_mf = np.zeros((agent.state_dim))
        self.u_s = np.zeros((agent.control_dim))

        # need to access the agent's blackboard in order to construct mean field vectors

        
        list_of_methods=['L_l', 'L_l_q_dot','L_mf_q_dot','H_l','H_l_nou','H_l_u','q_rhs_H_l_nou',
                 'p_rhs_H_l_nou','q_rhs_H_l_u','p_rhs_H_l_u','q_rhs_H_l','p_rhs_H_l',
                 'qp_rhs_H_l', 'H_mf_nou','H_mf_u','q_rhs_H_mf_u','p_rhs_H_mf_u','q_rhs_H_mf_nou',
                 'p_rhs_H_mf_nou','p_rhs_H_mf','q_rhs_H_mf','qp_rhs_H_mf','qp_rhs','u_rhs']

        # create dictionary of correct inputs, mapping method name to inputs
        inputs = {'L_l': (self.q_s, self.p_l),
                 }

        # create list of correct dimensions for outputs
        outputs = {'L_l': (), # remember we check these using np.shape, so if it returns a scalar, then you need to call np.shape(<scalar output>) == ()
                  } 

    def check_All(self):
        '''
        Inputs:
            agent:
        Outputs: 
            method_report: string describing the issues, if any, with each method.  There are four possible outcomes (not mutually exclusive:  No error, missing, incorrect inputs, incorrect outputs.
        ''' 
        #TODO: figure out a clever way to test inputs, rather than just artificially constructing the correct dimensions.  Remember you can use some of the universal ("universal" meaning all agents have them - they are not the same across all agents) attributes of the batteryAgent, such as q_s_0, p_l_0, etc.
        for i in dir(self):
            result = getattr(self, i)
            if i.startswith('check') and not i.startswith('check_All') and hasattr(result, '__call__'):
                
               result()

#    def check_validate_dimensions(self):
#        pass
#
#    def check_L_l(self):
#        agent = self.agent
#        # check existence
#        if 'L_l' in dir(agent):
#            L_l_callable = getattr(agent, 'L_l')
#            if hasattr(L_l_callable, '__call__'):
#                # call the method
#                # check the output, and report NA if the method doesn't exist
#                output = L_l_callable()
#                self.check_output()
#            else:
#                existence = False
#        else:
#            existence = False
#        return existence, output
#
#        # check dimensions of output
#         L_l = agent.L_l(self.q_s, self.q_s_dot, agent.u_s)
#       
#        if isinstance(L_l, list):
#            output_result = 'correct'
#        else:
#            output_result = 'incorrect output dimensions'
#        
#    def check_L_l_q_dot(self):
#        agent = self.agent
#        L_l_q_dot = agent.L_l_q_dot(self.q_s, self.q_s_dot, agent.u_s)
#
#    def check_L_mf_q_dot(self):
#        agent = self.agent
#        L_mf_q_dot = agent.L_mf_q_dot(agent.q_mf, agent.q_mf_dot, agent.u_mf)
#
#    def check_H_l(self):
#        agent = self.agent
#        H_l = agent.H_l(self.q_s, agent.p_l, agent.u_s, lambda_l)
#
#    def check_H_l_nou(self):
#        agent = self.agent
#        H_l_nou = agent.H_l_nou(self.q_s, agent.p_l, lambda_l)
#
#    def check_H_l_u(self):
#        agent = self.agent
#        H_l_u = agent.H_l_u(self.q_s, agent.p_l)
#
#    def check_q_rhs_H_l_nou(self):
#        agent = self.agent
#        q_rhs_H_l_nou = agent.q_rhs_H_l_nou(self.q_s, agent.p_l, lambda_l)
#  
#    def check_p_rhs_H_l_nou(self):
#        agent = self.agent
#        p_rhs_H_l_nou = agent.p_rhs_H_l_nou(self.q_s, agent.p_l, lambda_l)
#
#    def check_q_rhs_H_l_u(self):
#        agent = self.agent
#        q_rhs_H_l_u = agent.q_rhs_H_l_u(self.q_s, agent.p_l)
#
#    def check_p_rhs_H_l_u(self, q_s, p_l):
#        pass
#    def check_q_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
#        pass
#    def check_p_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
#        pass
#    def check_qp_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
#        pass
#    def check_H_mf_nou(self, q_mf, p_mf, u_mf):
#        pass
#    def check_H_mf_u(self, q_mf, p_mf, u_mf):
#        pass
#    def check_q_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
#        pass
#    def check_p_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
#        pass
#    def check_q_rhs_H_mf_nou(self, q_mf, p_mf):
#        pass
#    def check_p_rhs_H_mf_nou(self, q_mf, p_mf):
#        pass
#    def check_p_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
#        pass
#    def check_q_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
#        pass
#    def check_qp_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
#        pass
#    def check_qp_rhs(self, t, qp_vec, **kwargs):
#        pass
#    def check_u_rhs(self, t, u_vec, **kwargs):
#        pass 
