# Jordan Makansi
# 11/18/18

import numpy as np
import scipy as sp
import ode
from sliding_window import *
from  blackboard import *


class Synchronizer:
    
    def __init__(self, agents, blackboard):
        self.agents = agents  # list of all agents. list with elements of class Agent
        self.bb = blackboard  # instance of class blackboard
        
        for agent in agents:
            self.bb.update_q_p_u_dict(agent)
            
        # add Synchronizer as to each agent
        for agent in agents:
            agent.sync = self
    
    def synchronize(self):
        # run synchronization by visiting each agent and running propagation
        for agent in self.agents:
            '''     
            1) run synchronized propagation - I think we only need one Agent instead of SlidingWindow now

            For each of the above 2 steps:
                - create sliding window instance
                - call "propagate_dynamics" on the sliding window instance

            get quenched values from blackboard
            '''
            # run propagation with keyword arguments state_dim_l, state_dim_mf
            q_ls_bars, p_ls_bars, p_mfs_bars, u_bars, windows = sliding_window(agent)

    def H_mf(self, q_mf, p_mf, u_mf, agent):
        # agent is an object of class agent.  It's the agent for which we are constructing H_mf.
        # start with H_mf = H_mf_nou, and then add the H_mf_u's

        H_mf = agent.H_mf_nou(q_mf, p_mf, u_mf)
        H_mf_u = agent.H_mf_u(q_mf, p_mf, u_mf)
        H_mf = H_mf + H_mf_u
        
        return H_mf
    
