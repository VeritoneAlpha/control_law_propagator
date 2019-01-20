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
    def check_All(self, agent):
        '''
        Inputs:
            agent:
        Outputs: 
            method_report: string describing the issues, if any, with each method.  There are four possible outcomes (not mutually exclusive:  No error, missing, incorrect inputs, incorrect outputs.
        ''' 
        for i in dir(self):
            result = getattr(self, i)
            if i.startswith('check') and not i.startswith('checkAll') and hasattr(result, '__call__'):
                result()

    def check_validate_dimensions(self):
        pass
    def check_L_l(self, q_s, q_s_dot, u_s):
        pass
    def check_L_l_q_dot(self, q_s, q_s_dot, u_s):
        pass
    def check_L_mf_q_dot(self, q_mf, q_mf_dot, u_mf):
        pass
        def L_l_q_dot_building1(q_mf, q_mf_dot, u_mf):
            pass
        def L_l_q_dot_building2(q_mf, q_mf_dot, u_mf):
            pass
    def check_H_l(self, q_s, p_l, u_s, lambda_l):
        pass
    def check_H_l_nou(self, q_s, p_l, lambda_l):
        pass
    def check_H_l_u(self, q_s, p_l):
        pass
    def check_q_rhs_H_l_nou(self, q_s, p_l, lambda_l):
        pass
    def check_p_rhs_H_l_nou(self, q_s, p_l, lambda_l):
        pass
    def check_q_rhs_H_l_u(self, q_s, p_l):
        pass
    def check_p_rhs_H_l_u(self, q_s, p_l):
        pass
    def check_q_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        pass
    def check_p_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        pass
    def check_qp_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
        pass
    def check_H_mf_nou(self, q_mf, p_mf, u_mf):
        pass
    def check_H_mf_u(self, q_mf, p_mf, u_mf):
        pass
    def check_q_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        pass
    def check_p_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
        pass
    def check_q_rhs_H_mf_nou(self, q_mf, p_mf):
        pass
    def check_p_rhs_H_mf_nou(self, q_mf, p_mf):
        pass
    def check_p_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        pass
    def check_q_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        pass
    def check_qp_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
        pass
    def check_qp_rhs(self, t, qp_vec, **kwargs):
        pass
    def check_u_rhs(self, t, u_vec, **kwargs):
        pass 
