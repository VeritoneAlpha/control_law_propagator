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
        ### define inputs
        qpu_vec = self.Agent2.qpu_vec
        #t_start = 
        #t_end = 
        sliding_window_instance = self.Agent2
        q_mf = construct_mf_vectors(self.Agent2)
        result_propagate_q_p = propagate_q_p(qpu_vec, t_start, t_end, sliding_window_instance, q_mf, u_mf)
   

def suite_test():
    """
        Gather all the tests from this module in a test suite.
    """
    unittest.main(verbosity=2)
    suite = unittest.TestSuite()
    suite.addTest(FunctionalityTestCase('test_propagate_q_p'))
   # suite.addTest(FunctionalityTestCase('test_demo1_chatter_optimizer'))
   # suite.addTest(FunctionalityTestCase('test_demo1_edgecht_integ_init'))
   # suite.addTest(FunctionalityTestCase('test_demo1_edgecht_integ_exe'))

    return suite


if __name__ == '__main__':
   
    mySuite=suite_test()
      
    #runner=unittest.TextTestRunner()
    #runner.run(mySuit)
    
    result = unittest.result.TestResult()
    mySuite.run(result)
    print result
