#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:16:51 2018
This file is for testing the CDI control law propagator.
"""

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
        test example:
            H(q,p,u)=-sum(i:1~3)(alpha_i*q_i^2+beta_i*q_i*p_i)+b1*u1*q1*p2+b2*u2*q2*p3+b3*p1*q3*(u1-u2)
            HD(q,p)=-a1*q1^2-a2*q2^2+a3*q3^2+c1*p1*q1+c2*p2*q2+c3*p3*q3
            Q=[H-HD]^2
        '''
        parBeta = np.array([0.1, 0.15, 0.01])
        parB = np.array([0.02, 0.05, 0.01]) #np.array([0.02, 0.05, 0.1])
        parAlpha = np.array([0.3, 0.8, 0.1])
        parA = np.array([1.5, 0.5, 0.1])
        parC = np.array([0.1, 0.01, 0.05]) ##np.array([0.1, 0.01, 0.5])
        
        class SlidingWindowExample(sliding_window.SlidingWindow):
            '''
            Implementation of SlidingWindow abc
            '''
            def qp_rhs(self, t, qp_vec, **kwargs): # Shen - when customize problem, how to define and pass parameters of differential equations?
                p = qp_vec[:len(qp_vec)/2]
                q = qp_vec[len(qp_vec)/2:]
                u = kwargs['u_0']
                # for q-dot=dH/dp
                '''
                parBeta = np.array([0.1, 0.15, 0.01])
                parB = np.array([0.02, 0.05, 0.1])
                parAlpha = np.array([0.3, 0.8, 0.1])
                '''
                
                q_dot =  np.zeros(np.shape(q))
                q_dot[0] = -parBeta[0]*q[0]+parB[2]*(u[0]-u[1])*q[2]
                q_dot[1] = -parBeta[1]*q[1]+parB[0]*u[0]*q[0]
                q_dot[2] = -parBeta[2]*q[2]+parB[1]*u[1]*q[1]

                # for p-dot=- dH/dq
                p_dot =  np.zeros(np.shape(q))
                p_dot[0] = 2.0*parAlpha[0]*q[0]+parBeta[0]*p[0]-parB[0]*u[0]*p[1]
                p_dot[1] = 2.0*parAlpha[1]*q[1]+parBeta[1]*p[1]-parB[1]*u[1]*p[2]
                p_dot[2] = 2.0*parAlpha[2]*q[2]+parBeta[2]*p[2]-parB[2]*(u[0]-u[1])*p[0]
                
                return np.concatenate([q_dot, p_dot])
              
            def u_rhs(self, t, u_vec, **kwargs):
                qp_vec = kwargs['qp_vec']
                p = qp_vec[:len(qp_vec)/2]
                q = qp_vec[len(qp_vec)/2:]
                Gamma = kwargs['Gamma']
                # for u-dot=-Gamma*d(Q_dot)/du
                qp_dot = self.qp_rhs(t, qp_vec, u_0=u_vec)
                q_dot = qp_dot[len(qp_vec)/2:]
                p_dot = qp_dot[:len(qp_vec)/2]
                H, HD = self.h_hd(q, p, u_vec)
                #H = -parAlpha.dot(q*q)-parBeta.dot(q*p)+parB[0]*u_vec[0]*q[0]*p[1]+parB[1]*u_vec[1]*q[1]*p[2]+parB[1]*(u_vec[0]-u_vec[1])*q[2]*p[0]
                #HD = -parA[0]*q[0]*q[0]-parA[1]*q[1]*q[1]+parA[2]*q[2]*q[2]+parC.dot(q*p)
                tmpTerm = (2.0*parA[0]*q[0]-parC[0]*p[0])*q_dot[0] +\
                          (2.0*parA[1]*q[1]-parC[1]*p[1])*q_dot[1] +\
                          (-2.0*parA[2]*q[2]-parC[2]*p[2])*q_dot[2] -\
                          parC[0]*q[0]*p_dot[0] - parC[1]*q[1]*p_dot[1] - parC[2]*q[2]*p_dot[2]
                u_dot = np.zeros(np.shape(u_vec))
                u_dot[0] = -Gamma*(2.0*(parB[0]*q[0]*p[1]+parB[2]*q[2]*p[0])*tmpTerm +\
                           2.0*(H-HD)*((2.0*parA[0]*q[0]-parC[0]*p[0])*parB[2]*q[2] +\
                           (2.0*parA[1]*q[1]-parC[1]*p[1])*parB[0]*q[0] +\
                           parC[0]*q[0]*parB[0]*p[1] + parC[2]*q[2]*parB[2]*p[0] ))
                u_dot[1] = -Gamma*(2.0*(parB[1]*q[1]*p[2]-parB[2]*q[2]*p[0])*tmpTerm +\
                           2.0*(H-HD)*(-(2.0*parA[0]*q[0]-parC[0]*p[0])*parB[2]*q[2] -\
                           (2.0*parA[2]*q[2]+parC[2]*p[2])*parB[1]*q[1] + parC[1]*q[1]*parB[1]*p[2] - \
                           parC[2]*q[2]*parB[2]*p[0]))
                
                return u_dot
        
            def h_hd(self, q, p, u_vec):
                '''
                evaluate Hamitonian and desired Hamiltonian
                '''
                H = -parAlpha.dot(q*q)-parBeta.dot(q*p)+parB[0]*u_vec[0]*q[0]*p[1]+parB[1]*u_vec[1]*q[1]*p[2]+parB[1]*(u_vec[0]-u_vec[1])*q[2]*p[0]
                HD = -parA[0]*q[0]*q[0]-parA[1]*q[1]*q[1]+parA[2]*q[2]*q[2]+parC.dot(q*p)
                return H, HD
        
            # Inputs for numerical propagator
            q_0 = np.array([0.1,0.2,0.3])
            p_0 = np.array([1.0, 2.0,3.0])
            u_0 = np.array([0.0, 0.0])#np.array([1.0, 1.1])
            qpu_vec = np.hstack([q_0, p_0, u_0])
            state_dim = 3
            Gamma = 0.01
            
            # Inputs for numerical integration
            integrateTol = 1e-6 %10**-3
            integrateMaxIter = 40
            
            # Inputs for sliding window
            t_0 = 0
            T = 0.1 #1.0
            K = 10
            t_terminal = 2
            n_s = 20

        self.sliding_window_instance = SlidingWindowExample()
 
    def test_sliding_window_simpleCase(self):
        # for testing: givn initial q,p,u, compute Q=(H-HD)^2
        iteration = 20
        objEvalQ = np.zeros(iteration+1)
        H, HD = self.sliding_window_instance.h_hd(self.sliding_window_instance.q_0, self.sliding_window_instance.p_0, 
                          self.sliding_window_instance.u_0)
        objEvalQ[0] = (H-HD)*(H-HD)

        
        # setup parameters, data and initial point for testing function
        # call sliding window         
        qpu_vec, q_bar, p_bar, u_bar, qs, ps, us = sliding_window.propagate_dynamics(self.sliding_window_instance)
        print q_bar
        print p_bar
        print u_bar
        
        # for testing: givn q_bar, p_bar, u_bar, compute Q=(H-HD)^2
        H, HD = self.sliding_window_instance.h_hd(q_bar, p_bar, u_bar)
        objEvalQ[1] = (H-HD)*(H-HD)
        
        i=1
        while i<iteration:
            
            #self.sliding_window_instance.Gamma = self.sliding_window_instance.Gamma*0.01
            self.sliding_window_instance.q_0 = q_bar
            self.sliding_window_instance.p_0 = p_bar
            self.sliding_window_instance.u_0 = u_bar
            self.sliding_window_instance.qpu_vec = np.hstack([q_bar, p_bar, u_bar])
            
            qpu_vec, q_bar, p_bar, u_bar, qs, ps, us = sliding_window.propagate_dynamics(self.sliding_window_instance)
            
            # simulation: implement u_bar
            '''
            q_0 = qs[self.sliding_window_instance.K-1]
            p_0 = ps[self.sliding_window_instance.K-1]
            u_0 = u_bar
            t_start = 0.0
            t_end = self.sliding_window_instance.T/2.0
            qp_vecs = sliding_window.propagate_q_p(q_0, p_0, u_0, t_start, t_end, self.sliding_window_instance)
            '''
            
            H, HD = self.sliding_window_instance.h_hd(q_bar, p_bar, u_bar)
            objEvalQ[i+1] = (H-HD)*(H-HD)
            
            print objEvalQ
            
            i+=1
        '''
        i=0
        while i<iteration:
        
            qpu_vec, qs, ps, us = sliding_window.propagate_dynamics(qpu_vec, sliding_window_instance)
            # qs, ps, and us will go to Mean Field somehow
    
            q_bar = apply_filter(qs, weights, weights_total)
            p_bar = apply_filter(ps, weights, weights_total)
            u_bar = apply_filter(us, weights, weights_total)
            
            i+=1
            
            q_bars.append(q_bar)
            p_bars.append(p_bar)
            u_bars.append(u_bar)
        '''    
        '''
        # compare with expected results 
        q_bars_expected = [np.array([0.])]
        p_bars_expected = [np.array([0.])]
        u_bars_expected = [np.array([0.])]
        actuals = [q_bar, p_bar, u_bar]
        expecteds = [q_bars_expected, p_bars_expected, u_bars_expected]
        for actual, expected in zip(actuals,expecteds):
            for actual_val, elm in zip(actual,expected):
                self.assertTrue(np.amax(abs(elm-actual_val))<1e-6, msg=None)
        '''

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
