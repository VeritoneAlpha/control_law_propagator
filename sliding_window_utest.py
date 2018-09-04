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
            def H_T_p(self, q,p,u):
                # for q-dot
                return np.zeros(np.shape(q))
            
            def H_T_q(self, q,p,u):
                # for p-dot
                return np.zeros(np.shape(p))
                
            def Q_u(self, q,p,u):
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
        
        self.sliding_window_instance = SlidingWindowExample()
 
    def test_sliding_window_simpleCase(self):
        # setup parameters, data  and initial point for testing function

        # call sliding window 
        q_bars, p_bars, u_bars = sliding_window.sliding_window(self.sliding_window_instance)
        print q_bars
        print p_bars
        print u_bars
        # compare with expected results 
        q_bars_expected = [np.array([0.]), np.array([0.])]
        p_bars_expected = [np.array([0.]), np.array([0.])]
        u_bars_expected = [np.array([0.]), np.array([0.])]
        actuals = [q_bars, p_bars, u_bars]
        expecteds = [q_bars_expected, p_bars_expected, u_bars_expected]
        for actual, expected in zip(actuals,expecteds):
            for actual_val, elm in zip(actual,expected):
                self.assertTrue(np.amax(abs(elm-actual_val))<1e-6, msg=None)
     

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
    unittest.main(verbosity=2) 
    mySuite=suite_test()
      
    result = unittest.result.TestResult()
    mySuite.run(result)
    print result
