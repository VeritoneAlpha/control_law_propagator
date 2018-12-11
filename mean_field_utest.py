import sys
sys.path.insert(0, './cdi-edge-controller')
import numerical_propagator as prp

import unittest   
import numpy as np

class FunctionalityTestCase(unittest.TestCase):
    def setUp(self):
        # define a sliding_window object
        pass
    
    def test_propagate_q_p(self):
        '''
        unittest for propagate_q_p
        '''
        
        ### define inputs
        qpu_vec = np.array([])
        t_start  = 0.1     # Ohms    
        t_end = 1.5 # Shen - should it be tCdiBkt?    
        sliding_window_instance = 100000.0 
        q_mf  = 1.0    
        u_mf
   
    def test_demo1_chatter_optimizer(self):
        '''
   self.assertTrue(np.amax(abs(chtOptimalAlpha.values-expOptimalAlpha))<1e-6, msg=None)       

    def test_demo1_edgecht_integ_exe(self):
        '''
        test regular of running demo1_chatter_optimizer, i.e. execute both chattering optimizer and PAE
        '''
        ### prepare input to demo1_chatter_optimizer
        demo1Par = {'dependForceParV0': 10.0, 'R_s_max': 1000, 'R_s_min': 0.01, 'q_B_min': 50.0, 
                    'q_B_max': 130.0, 'independVolVu': 14.4, 'inductanceL': 0.1, 'delta': 0.1, 
                    'dependForceParK': 1.0, 'capacitorC': 200.0}
        
        edgeChtInit = {'demo1Init':[],'incrementalCovWInit':[],'stateDim':[],'controlDim':[],
                       'chtTrackParQ':[], 'chtCrossParS':[], ' chtLinearParRu':[], 
                       'chtSteps':[],'chtTDelta':[], 'chtULevels':[], 'chtUVectorOfLevel':[]}

        ### bucket divider setup for chattering
        edgeChtInit['chtBktDivider'] = int(2) # int(10)
        edgeChtInit['paeBktMultiplier'] = int(4)#int(10)
        
        ### define dimensions of the state, control and observation for demo-1 edge controller model
        edgeChtInit['stateDim'] = int(5)
        edgeChtInit['controlDim']  = int(1)   
        
        ### setup variabl for storing initial state for constructing demo-1 edge controller model with LQ tracking
        edgeChtInit['demo1Init'] = {'Time':[],'iInductor':[], 'vBDependForce': [],'qBDependForce':[],
                                    'powerRateDependForce': [], 'vCCapacitor':[], 'control':[]}
        
        ### chattering weight matics setup
        ###### chtTrackParQ   :  chattering tracking weight matrix for state vector (i.e. delta_x)
        edgeChtInit['chtTrackParQ'] = np.diag(np.array([1e-1,1e-1,1e-1,0.5,1e-1]))
        ###### chtCrossParS   :  chattering cross term weight matrix for state and control vectors
        edgeChtInit['chtCrossParS'] = np.array([[ 1e-5],[ 1e-5],[ 1e-5],[ 1e-5],[ 1e-5]])
        ###### chtLinearParRu :  chattering linear term weight matrix for control vector
        edgeChtInit['chtLinearParRu'] = np.array([0.0])
        
        ### define number of sub-interval within chattering time interval
        edgeChtInit['chtSteps'] = int(2) #int(5)
                
        # demo-1 edge controller design: PAE 
        paeW = 1e-1*np.diag(np.array([5e-3, 7e-5, 4e-2, 1e-1, 8e-4])) # 1e-1*np.diag(np.array([5e-3, 7e-5, 4e-2, 1e-1, 8e-4]))
        paeVarianceParA = np.array([[1e-8, 2e-4, 1e-8, 1e-8, 2e-4], \
                                    [3e-6, 0, 5e-7, 1e-8, 1e-8], \
                                    [1e-2, 1e-8, 1e-8, 1e-8, 1e-8],\
                                    [3e-4,2e-2,1e-6,1e-8,2e-2],\
                                    [2e-4, 1e-8, 1e-8, 1e-8, 1e-5]])
        paeVarianceParB = np.array([[1e-8], [1e-8], [1e-8], [1e-8], [1e-1]])        
        paeVarianceParF = np.array([5e-3, 7e-5, 4e-2, 1e-1, 8e-4]) 
        
        paeVarianceInit = {'parA':paeVarianceParA, 'parB':paeVarianceParB, 'parF':paeVarianceParF, 'paeW':paeW}
        
        edgeChtInit['demo1Init'] = {'powerRateDependForce': 115.251, 'qBDependForce': 120.1011, 
                   'control': 0.01, 'vBDependForce': 14.7652, 'iInductor': -11.8446, 'Time': 10.0, 
                   'vCCapacitor': 13.942500000000001}

        ### initiate frequency design, chattering and PAE
        tCdiBkt = np.float(0.1)
        freqDesignCht, incChtStruct, paeStruct = dm1cht.demo1_edgecht_integ_init(demo1Par, edgeChtInit, 
                                                                          paeVarianceInit, tCdiBkt)
        

        stateAtStartT = np.array([ [-11.8384,   14.7624,  120.0524,  115.2516,   14.0323],
                                   [ -11.4618,   14.7653,  120.4763,  115.0545,   13.9432],
                                   [ -11.0676,   14.8035,  121.1263,  114.8369,   14.0573],
                                   [ -10.6929,   14.7699,  121.5406,  114.5919,   14.0389],
                                   [ -10.3026,   14.8663,  122.3157,  114.3296,   14.0082],
                                   [  -9.8894,   14.7778,  122.7988,  114.0477,   13.9982]])
        controlAtStartT = np.array([0.01, 99.999999999786354, 100.0, 100.0, 100.0, 100.0])

 
        ### setup tracking signals of state vector 
        zData = np.array([[  -8.61391798,   14.82156857, 124.1596918 ,  132.28990438, 13.92929328],
                              [  -8.61391798,   14.82156857, 124.1596918 ,  132.28990438, 13.92929328]])
        signalTrack = {'timeIndex': [], 'zTrack': zData, 'uTrack': np.array([[ 0.001],[ 0.001]])}    

        ### over future time interval [incChatter['tStart'], incChatter['tEnd']] with stepsize = freqDesignCht['tChtBkt']        
        tStart = 10.0
        for i in range(6):
    
            incChtStruct['incChatter']['tStart'] = tStart
            incChtStruct['incChatter']['tEnd'] = tStart + freqDesignCht['tChtBkt']
            
            timeIndex = np.zeros(incChtStruct['incChatter']['steps'])
            timeIndex[0] = incChtStruct['incChatter']['tStart']+incChtStruct['incChatter']['tDelta']
            for kk in range(1, incChtStruct['incChatter']['steps']):
                timeIndex[kk] = timeIndex[kk-1] + incChtStruct['incChatter']['tDelta']            
            signalTrack['timeIndex']= timeIndex

            # update chattering state and control at the beginning of chattering interval
            incChtStruct['incChatter']['stateAtStartT'] = stateAtStartT[i]
            incChtStruct['incChatter']['controlAtStartT'] = controlAtStartT[i]      
            
            # execute chattering and PAE for future time interval ['currentTime', 'futureTime]
            incChtStruct, chtOptimalControl, chtOptimalAlpha, paeStruct = \
                    dm1cht.demo1_edgecht_integ_exe(signalTrack, freqDesignCht, 
                                                       incChtStruct, paeStruct, demo1Par)
            # convert resistance into inverse of resistance (control in simulator)
            #uPastControl = 1.0/chtOptimalControl
            
            tStart = tStart + freqDesignCht['tChtBkt']

        


        # compare with expected results 
        expPaeParABF = np.array([ -9.78994052e-07,   9.98792290e+00,  -2.02156491e-06,
                                 7.64888848e-07,  -9.95893337e+00,  -7.26485390e-04,
                                 6.85699329e+00,  -1.04246752e-02,   0.00000000e+00,
                                -8.57551533e-03,  -6.75243984e-07,  -2.64700723e-05,
                                 2.47494045e-03,   1.31212797e-01,  -9.06866619e-01,
                                 3.47157497e-08,   2.12875385e-07,  -1.06478429e-07,
                                -1.07745555e-07,  -1.64549195e-05,   1.28706958e+01,
                                -4.21556937e+00,   1.58776085e+00,  -1.51588973e-01,
                                 3.55880228e-05,   1.94237626e+01,  -3.52277992e-01,
                                -3.16074456e+00,  -6.70001839e-03,   3.78345390e-06,
                                 1.02767651e-05,   2.54183410e-06,   7.13748019e-03,
                                -2.71650359e-03,  -4.01685637e-01])
    
        expParB = np.array([[ -7.26485390e-04],[  2.47494045e-03],[ -1.64549195e-05],[ -3.52277992e-01],[ -2.71650359e-03]]), 
        expParA = np.array([[ -9.78994052e-07,   9.98792290e+00,  -2.02156491e-06,
                                                  7.64888848e-07,  -9.95893337e+00],
                                               [ -1.04246752e-02,   0.00000000e+00,  -8.57551533e-03,
                                                 -6.75243984e-07,  -2.64700723e-05],
                                               [ -9.06866619e-01,   3.47157497e-08,   2.12875385e-07,
                                                 -1.06478429e-07,  -1.07745555e-07],
                                               [ -4.21556937e+00,   1.58776085e+00,  -1.51588973e-01,
                                                  3.55880228e-05,   1.94237626e+01],
                                               [ -6.70001839e-03,   3.78345390e-06,   1.02767651e-05,
                                                  2.54183410e-06,   7.13748019e-03]]) 
        expParF = np.array([  6.85699329,   0.1312128 ,  12.87069578,  -3.16074456,  -0.40168564])
        
        expOptimalControl = np.array([[0.01],[0.01],[0.01]])
        expOptimalAlpha = np.array([[  3.388900e-12,  1.0],[  2.260192e-12,  1.0]])
        self.assertTrue(max(abs(chtOptimalControl.index-np.array([10.250, 10.275,  10.3])))<1e-6, msg=None)
        self.assertTrue(np.amax(abs(chtOptimalControl.values-expOptimalControl))<1e-6, msg=None)       
        self.assertTrue(max(abs(chtOptimalAlpha.index-np.array([10.275,  10.3])))<1e-6, msg=None)
        self.assertTrue(np.amax(abs(chtOptimalAlpha.values-expOptimalAlpha))<1e-6, msg=None)  
        self.assertTrue(np.amax(abs(paeStruct['paeParAndCov']['paeParABF']-expPaeParABF))<1e-6, msg=None) 
        self.assertTrue(np.amax(abs(incChtStruct['incStateDyn']['parB']-expParB))<1e-6, msg=None) 
        self.assertTrue(np.amax(abs(incChtStruct['incStateDyn']['parA']-expParA))<1e-6, msg=None) 
        self.assertTrue(np.amax(abs(incChtStruct['incStateDyn']['parF']-expParF))<1e-6, msg=None) 

    def test_demo1_edgecht_integ_init(self):
        '''
        test demo1_edgecht_integ_init, i.e. initialize demo-1 model for integrating chattering with PAE
        '''
        ### demo-1 circuit parameters
        demo1Par = {'inductanceL':np.float(20),'capacitorC':np.float(60),'independVolVu':np.float(48),
                    'dependForceParV0':np.float(47),'dependForceParK':np.float(0.5)}
        R_s_max = 10000.0  # Ohms
        R_s_min = 0.1     # Ohms    
        tCdiBkt = 10.0 # Shen - should it be tCdiBkt?    
        q_B_max = 10000.0 
        q_B_min = 10.0    
        demo1Par['R_s_max'] = R_s_max
        demo1Par['R_s_min'] = R_s_min
        demo1Par['delta'] = tCdiBkt
        demo1Par['q_B_max'] = q_B_max
        demo1Par['q_B_min'] = q_B_min
           
        # user provided info for initializing demo-1 model for edge controller with chattering
        edgeChtInit = {'demo1Init':[],'incrementalCovWInit':[],'stateDim':[],'controlDim':[],
                       'chtTrackParQ':[], 'chtCrossParS':[], ' chtLinearParRu':[], 
                       'chtSteps':[],'chtULevels':[], 'chtUVectorOfLevel':[]}
    
        ### bucket divider setup for chattering
        edgeChtInit['chtBktDivider'] = int(10) # int(2)
        edgeChtInit['paeBktMultiplier'] = int(5)#int(10)
        
        ### define dimensions of the state, control and observation for demo-1 edge controller model
        edgeChtInit['stateDim'] = int(5)
        edgeChtInit['controlDim']  = int(1)   
        
        ### setup variabl for storing initial state for constructing demo-1 edge controller model with LQ tracking
        edgeChtInit['demo1Init'] = {'Time':10.0,'iInductor':-11.8608, 'vBDependForce': 14.7924,'qBDependForce':119.8246,
                                        'powerRateDependForce': 115.2518, 'vCCapacitor':14.0372, 'control':0.01}
            
        ### chattering weight matics setup
        ###### chtTrackParQ   :  chattering tracking weight matrix for state vector (i.e. delta_x)
        edgeChtInit['chtTrackParQ'] = np.diag(np.array([1e-1,1e-1,1e-1,0.5,1e-1]))
        ###### chtCrossParS   :  chattering cross term weight matrix for state and control vectors
        edgeChtInit['chtCrossParS'] = np.array([[ 1e-5],[ 1e-5],[ 1e-5],[ 1e-5],[ 1e-5]])
        ###### chtLinearParRu :  chattering linear term weight matrix for control vector
        edgeChtInit['chtLinearParRu'] = np.array([0.0])
        
        ### define number of sub-interval within chattering time interval
        edgeChtInit['chtSteps'] = int(3)
        #edgeChtInit['chtTDelta'] = 0.5
        ### provides the values of control vector at each level
        chtULevels = int(2)
        edgeChtInit['chtUVectorOfLevel'] = np.zeros([chtULevels,edgeChtInit['controlDim']])
        edgeChtInit['chtUVectorOfLevel'][0,0] = 1e-5 #1e-4
        edgeChtInit['chtUVectorOfLevel'][1,0] = 1e-1   
    
        # demo-1 edge controller design: PAE 
        paeW = 1e-1*np.diag(np.array([5e-3, 7e-5, 4e-2, 1e-1, 8e-4])) # 1e-1*np.diag(np.array([5e-3, 7e-5, 4e-2, 1e-1, 8e-4]))
        paeVarianceParA = np.array([[1e-8, 2e-4, 1e-8, 1e-8, 2e-4], \
                                    [3e-6, 0, 5e-7, 1e-8, 1e-8], \
                                    [1e-2, 1e-8, 1e-8, 1e-8, 1e-8],\
                                    [3e-4,2e-2,1e-6,1e-8,2e-2],\
                                    [2e-4, 1e-8, 1e-8, 1e-8, 1e-5]])
        paeVarianceParB = np.array([[1e-8], [1e-8], [1e-8], [1e-8], [1e-1]])        
        paeVarianceParF = np.array([5e-3, 7e-5, 4e-2, 1e-1, 8e-4])     
        paeVarianceInit = {'parA':paeVarianceParA, 'parB':paeVarianceParB, 'parF':paeVarianceParF, 'paeW':paeW}
    
        tCdiBkt = np.float(10.0) 
        
        # call demo1_edgecht_integ_init
        freqDesignCht, incChtStruct, paeStruct  = dm1cht.demo1_edgecht_integ_init(demo1Par, edgeChtInit, 
                                                                           paeVarianceInit, tCdiBkt)


def suite_test():
    """
        Gather all the tests from this module in a test suite.
    """
    suite = unittest.TestSuite()
    suite.addTest(FunctionalityTestCase('test_demo1_chatter_init'))
    suite.addTest(FunctionalityTestCase('test_demo1_chatter_optimizer'))
    suite.addTest(FunctionalityTestCase('test_demo1_edgecht_integ_init'))
    suite.addTest(FunctionalityTestCase('test_demo1_edgecht_integ_exe'))

    return suite


if __name__ == '__main__':
   
    #mySuite=suite_vector()
    mySuite=suite_test()
      
    #runner=unittest.TextTestRunner()
    #runner.run(mySuit)
    
    result = unittest.result.TestResult()
    mySuite.run(result)
    print result
