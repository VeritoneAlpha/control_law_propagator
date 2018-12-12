# Jordan Makansi
# 11/18/18

import sys
sys.path.insert(0, './cdi-edge-controller')
import numerical_propagator as prp
import unittest   
import numpy as np
from agents_for_testing import *
from blackboard import * 

class FunctionalityTestCase(unittest.TestCase):
    def setUp(self):
        # define blackboard and agent
        self.bb=Blackboard() 
        # define a sliding_window object (maybe do this in another file)
        self.Agent2=Agent(self.bb, state_indices=[1], control_indices=[1])
        pass
    
    def test_propagate_q_p(self):
        '''
        unittest for propagate_q_p
        '''
        ### define inputs
        qpu_vec = self.Agent2.qpu_vec
        t_start  = 0.1     # Ohms    
        t_end = 1.5 # Shen - should it be tCdiBkt?    
        sliding_window_instance = 100000.0 
        q_mf  = 1.0    
   

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
