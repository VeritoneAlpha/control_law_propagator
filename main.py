# Jordan Makansi
# 12/13/18


import yaml

# import the files where the Agent classes are defined
# for now, use the testing agents but in the future use actual agent files
from agents_for_testing import *


def sliding_window(sliding_window_instance):
    ''' 
    This method runs the propagation for a single agent.  Corresponding to the flow chart it runs:
        - Read from blackboard to get the following observation measured at time t_0, and onwards
        - construct quenched mean field for Hamiltonian agent i
        - Setup initial conditions for L_MF and p_MF
        - Construct agent synchronized Hamiltonian and partial derivatives
    
    Inputs:
        sliding_window_instance (instance of user-defined class which inherits SlidingWindow): object defining the dynamics, and the initial conditions and parameters for numerical integration/propagation. 
        The only input is sliding_window_instance, but we use the following attributes of the sliding_window_instance:
            t_0 (np.float): Initial time to start propagating dynamics
            T (np.float): End time of propagating dynamics
            qpu_vec (1D np.array of dimension self.state_dim*3+control_dim): Most current values for local qpu_vec containing q_s, p_l, p_mf, u_s, concatenated in one array
            state_dim (int): number of states
            Gamma (float): algorithmic parameter for Riemann descent algorithm
            t_terminal (int): time marking termination of control law propagator algorithm
    Outputs:
         q_ls_bars (list of np.arrays): implemented state/costate/control values for entire propagator.
         p_ls_bars (list of np.arrays, each of dimension self.state_dim): implemented local costate values for entire propagation.
         p_mfs_bars (list of np.arrays, each of dimension self.state_dim): implemented mean field costate values for entire propagation.
         u_bars (list of np.arrays, each of dimension self.control_dim): implemented control values for entire propagation.
         windows (list of lists): windows for each vector of bar values to be implemented
    '''
    
    t_0, T, K, state_dim,t_terminal = sliding_window_instance.t_0, sliding_window_instance.T, sliding_window_instance.K,  sliding_window_instance.state_dim, sliding_window_instance.t_terminal
    q_ls_bars, p_ls_bars, p_mfs_bars, u_bars, windows = [], [], [], [], []
    q_ls_dot_bars, p_ls_dot_bars, p_mfs_dot_bars, p_mfs_dot_bar, u_dot_bars =[], [], [], [], []
    t = t_0 # wall clock time
    
    # Read from blackboard to get the following observations measured at time t_0
    q_s_0, q_s_dot_0, u_s_0 = construct_local_vectors(sliding_window_instance)
    q_mf, q_mf_dot, u_mf = construct_mf_vectors(sliding_window_instance)

    # now pick out the individual states that we need to make q_s and q_s_dot
    qpu_vec = sliding_window_instance.qpu_vec
    state_dim = sliding_window_instance.state_dim
    # a note on q_s_dot - normally I understand that this would come from the sensors, ...
    # ...but for now get it from q_mf_dot from the blackboard, and just select if from the states that pertain to this agent
    # construct quenched mean field for Hamiltonian agent i
    # this happens inside of the class Synchronize method
    
    # If control is physical, then we should use the physical value here for initial condition
    # IF not, then we can use the average u, averaged over the previous window.
    # For now, use the value from the blackboard
    q_l_D_dot_0 = q_s_dot_0
    q_l_D_0 = q_s_0

    # set initial conditions using values from blackboard retrieved above
    # set initial conditions for local Hamiltonian of agent i
    p_l_0 = sliding_window_instance.L_l_q_dot(q_s_0, q_s_dot_0, u_s_0) # compute using Dirac compatibility
    p_l_D_0 = sliding_window_instance.L_l_D_q_Dot(q_l_D_0, q_l_D_dot_0) # compute using Dirac compatibility
    H_l_D_0 = sliding_window_instance.H_l_D(q_l_D_0, p_l_D_0)
    
    # setup initial condition for p_mf
    p_mf_0 = sliding_window_instance.L_mf_q_dot(q_mf, q_mf_dot, u_mf)
    
    # now construct qpu_vec 
    sliding_window_instance.qpu_vec = np.concatenate([q_s_0, p_l_0, p_mf_0, u_s_0]) # fill in with blackboard values for q and u, but for p, must be computed
    # Construct local Hamiltonian of agent i
    lambdas = sliding_window_instance.compute_lambdas(q_s_0, p_l_0, u_s_0)

    while t < sliding_window_instance.t_terminal:
        # for the times, propagate_dynamics needs: t_0, T, and K.  T and K can come from the sliding_window_instance
        #...t_0 will be passed in.  t_0 is the start of the window.

        # this propagates a single window inside of propagate dynamics
        qpu_vec, q_ls_bar, p_ls_bar, p_mfs_bar, u_bar, q_ls, p_ls, p_mfs, us, q_ls_dot_bar, p_ls_dot_bar, p_mfs_dot_bar, p_mfs_dot_bar, u_dot_bar, window = propagate_dynamics(sliding_window_instance)
        # qs, ps, and us will go to Mean Field somehow
        
        q_ls_bars.append(q_ls_bar)
        p_ls_bars.append(p_ls_bar)
        p_mfs_bars.append(p_mfs_bar)
        u_bars.append(u_bar)
        q_ls_dot_bars.append(q_ls_dot_bar)
        p_ls_dot_bars.append(p_ls_dot_bar)
        p_mfs_dot_bars.append(p_mfs_dot_bar)
        u_dot_bars.append(u_dot_bar)
        windows.append(window)

        t+=1
    # update blackboard
    bb=sliding_window_instance.bb
    bb.update_q_p_u_dict(sliding_window_instance)
    bb.save_bar_values(sliding_window_instance, q_ls_bars, p_ls_bars, p_mfs_bars, u_bars, q_ls_dot_bars, p_ls_dot_bars, p_mfs_dot_bars, u_dot_bars) # these eventually need to go to simulink and MATLAB interface
    
    return q_ls_bars, p_ls_bars, p_mfs_bars, u_bars, windows
   

if __name__ == "__main__":
    # initialize blackboard
    bb = Blackboard()
    
    # read in config 
    configFileName = 'cdi_config_agent1.yml'
    with open(configFileName, 'r') as f:
        initCDIModelInfo = yaml.safe_load(f)
    f.close()

    # put values from each agent into the blackboard
    bb.q_p_u_dict = initCDIModelInfo['bbValues']
 
    # initialize agent, and populate its values using values in config
    state_indices=initCDIModelInfo['agentParam']['state_indices']
    control_indices=initCDIModelInfo['agentParam']['control_indices']

    q_s_0 = np.array(initCDIModelInfo['agentParam']['q_s_0'])
    p_l_0 = np.array(initCDIModelInfo['agentParam']['p_l_0'])
    p_mf_0 = np.array(initCDIModelInfo['agentParam']['p_mf_0'])
    u_s_0 = np.array(initCDIModelInfo['agentParam']['u_s_0'])
    q_s_dot_0 = np.array(initCDIModelInfo['agentParam']['q_s_dot_0'])
 
    agent = Agent1(bb, state_indices, control_indices, q_s_0=None, p_l_0=None, p_mf_0=None, u_s_0=None, q_s_dot=None, gamma=1, Gamma=1, name='', integrateTol=10**-5, integrateMaxIter=400, t_0=0, T=2, K=4, t_terminal=4, n_s=10) 
   
    ### initiate current time
    tc = tstart
    
    while tc < tend:
        q_ls_bars, p_ls_bars, p_mfs_bars, u_bars, windows = sliding_window(agent)
        # increment tc, etc
    

