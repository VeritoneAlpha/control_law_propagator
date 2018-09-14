#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 3 19:09:22 2018
This file is for testing the sliding window as part of the control law propagator.
"""

import unittest
import numpy as np
import scipy as sp
import ode
import sliding_window

class FunctionalTestCase(unittest.TestCase):
    '''
    test sliding window 
    '''
    def setUp(self):         
        '''
        define classes to be used in the tests below
        '''
        class SlidingWindowExample(sliding_window.SlidingWindow):
            '''
            Implementation of SlidingWindow abc
            '''
            def qp_rhs(self, t, qp_vec, **kwargs):
                dim = len(qp_vec)/4
                q = qp_vec[:dim]
        	p = qp_vec[dim:2*dim]
                q_D = qp_vec[2*dim:3*dim]
                p_D = qp_vec[3*dim:]
                u = kwargs['u_0']
                # for q-dot
                q_dot =  np.zeros(np.shape(p))
                # for p-dot
                p_dot =  np.zeros(np.shape(q))
                q_D_dot =  np.zeros(np.shape(p))
                p_D_dot =  np.zeros(np.shape(q))
                return np.concatenate([q_dot, p_dot, q_D_dot, p_D_dot])
              
            def u_rhs(self, t, u_vec, **kwargs):
                qp_vec = kwargs['qp_vec']
                dim = len(qp_vec)/4
                q = qp_vec[:dim]
        	p = qp_vec[dim:2*dim]
                q_D = qp_vec[2*dim:3*dim]
                p_D = qp_vec[3*dim:]
                Gamma = kwargs['Gamma']
                # for u-dot
                return -1*Gamma*np.zeros(np.shape(u_vec))
                
            # Inputs for numerical propagator
            q_0 = np.array([0])
            p_0 = np.array([0])
            q_D = np.array([0])
            p_D = np.array([0])
            u_0 = np.array([0])
            qpu_vec = np.hstack([q_0, p_0, q_D, p_D, u_0])
            state_dim = 1
            Gamma = 1
                   
            # Inputs for numerical integration
            integrateTol = 10**-3
            integrateMaxIter = 40
            
            # Inputs for sliding window
            t_0 = 0
            T = 2
            K = 1
            t_terminal = 2
            n_s = 10

        self.sliding_window_instance = SlidingWindowExample()
 
    def test_sliding_window_simpleCase(self):
        # setup parameters, data and initial point for testing function
        # call sliding window 
        qpu_vec, q_bar, p_bar, q_D_bar, p_D_bar,  u_bar, qs, ps, q_Ds, p_Ds, us = sliding_window.propagate_dynamics(self.sliding_window_instance)
        print q_bar
        print p_bar
        print u_bar
        # compare with expected results 
        q_bars_expected = [np.array([0.])]
        p_bars_expected = [np.array([0.])]
        q_D_bars_expected = [np.array([0.])]
        p_D_bars_expected = [np.array([0.])]
        u_bars_expected = [np.array([0.])]
        actuals = [q_bar, p_bar, q_D_bar, p_D_bar, u_bar]
        expecteds = [q_bars_expected, p_bars_expected, q_D_bars_expected, p_D_bars_expected, u_bars_expected]
        for actual, expected in zip(actuals,expecteds):
            for actual_val, elm in zip(actual,expected):
                self.assertTrue(np.amax(abs(elm-actual_val))<1e-6, msg=None)
   

def suite_test():
    """
        Gather all the tests from this module in a test suite.
    """
    suite = unittest.TestSuite()
    suite.addTest(FunctionalTestCase('test_sliding_window_simpleCase'))
    return suite


if __name__ == '__main__':
    unittest.main(verbosity=2) 
    mySuite=suite_test()
      
    result = unittest.result.TestResult()
    mySuite.run(result)
    print result
