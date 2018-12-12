# Jordan Makansi
# 11/18/18

import sys
sys.path.insert(0, './cdi-edge-controller')
import numerical_propagator as prp
import unittest   
import numpy as np
from agents_for_testing import *
from blackboard import * 
from synchronizer import *

class FunctionalityTestCase(unittest.TestCase):
    def setUp(self):
        # define blackboard and agent
        self.bb=Blackboard() 
        # define a sliding_window object (maybe do this in another file)
        self.Agent1=Agent1(self.bb, state_indices=[1], control_indices=[1])
        self.Agent2=Agent2(self.bb, state_indices=[1,2], control_indices=[1])
        # add agents to the blackboard
        self.bb.update_q_p_u_dict(self.Agent1)
        self.bb.update_q_p_u_dict(self.Agent2)
        # add the agents to the synchronizer
        self.agents=[self.Agent1, self.Agent2]
        self.sync=Synchronizer(self.agents, self.bb)
    
    def test_propagate_q_p(self):
        '''
        unittest for propagate_q_p
        '''
        self.Agent2.n_s = 2
        ### define inputs
        qpu_vec = np.array([0, 2, 0, 3, 0, 1, 0])
        t_start = 0.0
        t_end = 0.25
        sliding_window_instance = self.Agent2
        q_mf, u_mf = [0.0, 2.0], [0.0]
        result_qp_vecs, result_qp_dot_vecs = propagate_q_p(qpu_vec, t_start, t_end, sliding_window_instance, q_mf, u_mf)
        
        expected_result_qp_vecs =[np.array([0.        , 2.13314708, 0.        , 2.74185292, 0.        ,
       1.13314708]), np.array([0.        , 2.28402231, 0.        , 2.46597769, 0.        ,
       1.28402231])] 
        expected_result_qp_dot_vecs =[np.array([ 0.        ,  1.13314708, -0.        , -2.13314708,  0.        ,
        1.13314708]), np.array([ 0.        ,  1.28402231, -0.        , -2.28402231,  0.        ,
        1.28402231])] 

        self.assertTrue(np.amax(abs(result_qp_vecs[0] - expected_result_qp_vecs[0]))<1e-6, msg=None) 
        self.assertTrue(np.amax(abs(result_qp_vecs[1] - expected_result_qp_vecs[1]))<1e-6, msg=None) 
        self.assertTrue(np.amax(abs(result_qp_dot_vecs[0] - expected_result_qp_dot_vecs[0]))<1e-6, msg=None) 
        self.assertTrue(np.amax(abs(result_qp_dot_vecs[1] - expected_result_qp_dot_vecs[1]))<1e-6, msg=None) 

    def test_propagate_u(self):
        '''
        unittest for propagate_u
        '''
        ### define inputs
        qpu_vec = np.array([0, 2, 0, 3, 0, 1, 0])
        t_start = 0.0
        t_end = 0.25
        sliding_window_instance = self.Agent2
        q_mf, u_mf = [0.0, 2.0], [0.0]
        result_qp_vecs, result_qp_dot_vecs = propagate_q_p(qpu_vec, t_start, t_end, sliding_window_instance, q_mf, u_mf)
    

def suite_test():
    """
        Gather all the tests from this module in a test suite.
    """
    unittest.main(verbosity=2)
    suite = unittest.TestSuite()
    suite.addTest(FunctionalityTestCase('test_propagate_q_p'))
    return suite


if __name__ == '__main__':
   
    mySuite=suite_test()
      
    #runner=unittest.TextTestRunner()
    #runner.run(mySuit)
    
    result = unittest.result.TestResult()
    mySuite.run(result)
    print result
