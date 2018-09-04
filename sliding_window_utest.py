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
        pass
    
    def test_sliding_window_simpleCase(self):
        # setup parameters, data  and initial point for testing function
        def H_T_p(q,p,u):
            # for q-dot
            return np.zeros(np.shape(q))
        
        def H_T_q(q,p,u):
            # for p-dot
            return np.zeros(np.shape(p))
            
        def Q_u(q,p,u):
            # for u-dot
            return np.zeros(np.shape(u))
        # Inputs for numerical propagator
        q_0 = np.array([0])
        p_0 = np.array([0])
        u_0 = np.array([0])
        qpu_vec = np.hstack([q_0, p_0, u_0])
        state_dim = 1
        Gamma = 1
        
        # Inputs for numerical integration
        integrateTol = 10**-3
        integrateMaxIter = 40
        
        # Inputs for sliding window
        t_0 = 0
        T =  2
        K=1
        
        # inputs for sliding window 
        t_terminal = 2

        # call sliding window 
        q_bars, p_bars, u_bars = sliding_window.sliding_window(t_0, T, K, q_0, p_0, u_0, state_dim, Gamma, t_terminal)

        # compare with expected results 
        expectTime = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])
        expectOptControl = np.array([[0.010000,  0.030000],\
                                    [0.510645,  0.642798],\
        expectOptState = np.array([[0.100000, -0.300000,  1.000000],\
                                    [0.044646, -0.263027,  0.919384],\
                                    [-0.013033, -0.225661,  0.881214],\
                                   [-0.166108, -0.133343 , 0.389243]])

        self.assertTrue(np.amax(abs(lqtOptimalControl.values-expectOptControl))<1e-6, msg=None)
        self.assertTrue(np.amax(abs(lqtOptimalState.values-expectOptState))<1e-6, msg=None)
        self.assertTrue(max(abs(lqtOptimalControl.index-expectTime))<1e-12, msg=None)
        self.assertTrue(max(abs(lqtOptimalState.index-expectTime))<1e-12, msg=None)
        self.assertTrue(not failRiccati)
        self.assertTrue(not failStatePrpg)
       

def suite_test():
    """
        Gather all the tests from this module in a test suite.
    """
    suite = unittest.TestSuite()
    suite.addTest(FunctionalTestCase('test_sliding_window_simpleCase'))
    #suite.addTest(FunctionalTestCase('test_sliding_window_simpleCase'))
    #suite.addTest(FunctionalTestCase('test_sliding_window_simpleCase'))
    return suite


if __name__ == '__main__':
   
    mySuite=suite_test()
      
    result = unittest.result.TestResult()
    mySuite.run(result)
    print result
