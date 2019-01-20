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
    def checkAll(self, agent):
    def validate_dimensions(self):
    def L_l(self, q_s, q_s_dot, u_s):
    def L_l_q_dot(self, q_s, q_s_dot, u_s):
    def L_mf_q_dot(self, q_mf, q_mf_dot, u_mf):
        def L_l_q_dot_building1(q_mf, q_mf_dot, u_mf):
        def L_l_q_dot_building2(q_mf, q_mf_dot, u_mf):
    def H_l(self, q_s, p_l, u_s, lambda_l):
    def H_l_nou(self, q_s, p_l, lambda_l):
    def H_l_u(self, q_s, p_l):
    def q_rhs_H_l_nou(self, q_s, p_l, lambda_l):
    def p_rhs_H_l_nou(self, q_s, p_l, lambda_l):
    def q_rhs_H_l_u(self, q_s, p_l):
    def p_rhs_H_l_u(self, q_s, p_l):
    def q_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
    def p_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
    def qp_rhs_H_l(self, q_s, p_l, u_s, lambda_l):
    def H_mf_nou(self, q_mf, p_mf, u_mf):
    def H_mf_u(self, q_mf, p_mf, u_mf):
    def q_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
    def p_rhs_H_mf_u(self, q_mf, p_mf, u_mf):
    def q_rhs_H_mf_nou(self, q_mf, p_mf):
    def p_rhs_H_mf_nou(self, q_mf, p_mf):
    def p_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
    def q_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
    def qp_rhs_H_mf(self, q_mf, p_mf, u_mf, u_s):
    def qp_rhs(self, t, qp_vec, **kwargs):
    def u_rhs(self, t, u_vec, **kwargs):
