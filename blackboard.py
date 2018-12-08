# Jordan Makansi
# 11/18/18

import numpy as np
import scipy as sp
import ode
from numerical_propagator import *

class Blackboard:
    
    def __init__(self):
        '''
        values: list of integers state_indices state_indices
        q_p_u_dict is a dictionary which maps 'q', 'p', 'u', to a dictionary of index-value pairs: local values for q, p, and u for this agent.  
                        blackboard holds all of the most recent local values, e.g.
                        {'q_s': {'1':3, '2':0}, 'p_mf': {'1':3, '2': 2}, 'u': {'1': 0}}
                        It doesn't care which agent updated them most recently.  It only needs to know which values to update.
        q_p_u_dict initially will be filled.
        '''
        ## TODO:  _s should really be called _mf because it contains all of the states/controls.
        self.q_p_u_dict = {'q_s':{}, 'p_l':{}, 'p_mf':{}, 'u_s':{}, 'q_s_dot':{}}
        self.agents=[]
        
    def update_q_p_u_dict(self, agent):
        '''This method should be called after propagation of each agent
        Inputs:
            agent (instance of class Agent): this is the agent whose values we are updating
        Outputs:
            None.  This method just updates the attributes of the blackboard,
            just update the dictionary, agent_q_p_u_dict.
        '''
        # Determine which states pertain to this agent and replace the old values with new
        for state_ix in agent.state_indices:
            self.q_p_u_dict['q_s'][str(state_ix)] = agent.qpu_vec[:agent.state_dim][state_ix-1]
            self.q_p_u_dict['p_l'][str(state_ix)] = agent.qpu_vec[agent.state_dim:2*agent.state_dim][state_ix-1]
            self.q_p_u_dict['p_mf'][str(state_ix)] = agent.qpu_vec[2*agent.state_dim:3*agent.state_dim][state_ix-1]
            self.q_p_u_dict['q_s_dot'][str(state_ix)] = agent.q_s_dot[state_ix-1]
            
        for control_ix in agent.control_indices:
            self.q_p_u_dict['u_s'][str(control_ix)] = agent.qpu_vec[3*agent.state_dim:][control_ix-1]
        
       
        # add agent if not already added 
        if agent not in self.agents:
            self.agents.append(agent)


def construct_mf_vectors(sliding_window_instance):
    '''helper function to get  q_mf, q_mf_dot, u_mf (incorrectly called q_s, q_s_dot, u_s, in the q_p_u_dict)
    This method supplements the q_s vector inside of sliding_window_instance with the values from the blackboard so that q_mf contains values for all of the states, not just the local ones.
    '''
    # need to get the values for all states and controls in order to construct q_mf, q_mf_dot, and u_mf
    # get them from the blackboard
    # construct q_mf, q_mf_dot, and u_mf using values from the blackboard
    bb = sliding_window_instance.bb
    qpu_vec = sliding_window_instance.qpu_vec
    q_mf_shape = len(bb.q_p_u_dict['q_s'].items())
    u_mf_shape = len(bb.q_p_u_dict['u_s'].items())
    q_mf = np.zeros(q_mf_shape)  # this should be an array of values
    q_mf_dot = np.zeros(q_mf_shape)  # this should be an array of values
    u_mf = np.zeros(u_mf_shape) # this should be an array of values
    '''
    Get entire state vector from the blackboard, and then overwrite values with local values for the states that pertain to this agent
    '''
    # get the indices for ALL of the states in entire system, from blackboard
    for q_ix, q_val in bb.q_p_u_dict['q_s'].items():
        # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(q_ix) in sliding_window_instance.state_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_mf[int(q_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            q_mf[int(q_ix)-1] = q_val

    for u_ix, u_val in bb.q_p_u_dict['u_s'].items():
        # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(u_ix) in sliding_window_instance.control_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.control_indices)==int(u_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            u_mf[int(u_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            u_mf[int(u_ix)-1] = u_val

    for q_dot_ix, q_dot_val in bb.q_p_u_dict['q_s_dot'].items(): # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(q_dot_ix) in sliding_window_instance.state_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_dot_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_mf_dot[int(q_dot_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            q_mf_dot[int(q_dot_ix)-1] = q_dot_val

    return q_mf, q_mf_dot, u_mf


def construct_local_vectors(sliding_window_instance):
    '''helper function to get  q_s, q_s_dot, u_s by reading data from sensors (i.e. blackboard in this case).
     cannot give us p_mf or p_l becuase p is non-physical and must be computed.
    output:  
        q_s, q_s_dot, u_s for local agent
    '''
    bb = sliding_window_instance.bb
    qpu_vec = sliding_window_instance.qpu_vec
    q_s_shape = len(sliding_window_instance.state_indices)
    u_s_shape = len(sliding_window_instance.control_indices)
    q_s = np.zeros(q_s_shape)  # this should be an array of values
    q_s_dot = np.zeros(q_s_shape)  # this should be an array of values
    u_s = np.zeros(u_s_shape) # this should be an array of values
    # get the indices for this agent
    for q_ix, q_val in bb.q_p_u_dict['q_s'].items():
        # if this index does NOT PERTAIN to this agent, then pass 
        # if the index does PERTAIN to this agent, then fill in with value from blackboard
        if int(q_ix) in sliding_window_instance.state_indices:
            # fill in with value from blackboard
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_s[int(q_ix)-1] = qpu_vec[qpu_ix[0][0]]

    for u_ix, u_val in bb.q_p_u_dict['u_s'].items():
        if int(u_ix) in sliding_window_instance.control_indices:
            qpu_ix = np.where(np.array(sliding_window_instance.control_indices)==int(u_ix))
            u_s[int(u_ix)-1] = qpu_vec[qpu_ix[0][0]]

    for q_dot_ix, q_dot_val in bb.q_p_u_dict['q_s_dot'].items(): # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        if int(q_dot_ix) in sliding_window_instance.state_indices:
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_dot_ix))
            q_s_dot[int(q_dot_ix)-1] = qpu_vec[qpu_ix[0][0]]

    return q_s, q_s_dot, u_s


def construct_mf_vectors(sliding_window_instance):
    '''helper function to get  q_mf, q_mf_dot, u_mf (incorrectly called q_s, q_s_dot, u_s, in the q_p_u_dict)
    This method supplements the q_s vector inside of sliding_window_instance with the values from the blackboard so that q_mf contains values for all of the states, not just the local ones.
    '''
    # need to get the values for all states and controls in order to construct q_mf, q_mf_dot, and u_mf
    # get them from the blackboard
    # construct q_mf, q_mf_dot, and u_mf using values from the blackboard
    bb = sliding_window_instance.bb
    qpu_vec = sliding_window_instance.qpu_vec
    q_mf_shape = len(bb.q_p_u_dict['q_s'].items())
    u_mf_shape = len(bb.q_p_u_dict['u_s'].items())
    q_mf = np.zeros(q_mf_shape)  # this should be an array of values
    q_mf_dot = np.zeros(q_mf_shape)  # this should be an array of values
    u_mf = np.zeros(u_mf_shape) # this should be an array of values
    '''
    Get entire state vector from the blackboard, and then overwrite values with local values for the states that pertain to this agent
    '''
    # get the indices for ALL of the states in entire system, from blackboard
    for q_ix, q_val in bb.q_p_u_dict['q_s'].items():
        # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(q_ix) in sliding_window_instance.state_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_mf[int(q_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            q_mf[int(q_ix)-1] = q_val

    for u_ix, u_val in bb.q_p_u_dict['u_s'].items():
        # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(u_ix) in sliding_window_instance.control_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.control_indices)==int(u_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            u_mf[int(u_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            u_mf[int(u_ix)-1] = u_val

    for q_dot_ix, q_dot_val in bb.q_p_u_dict['q_s_dot'].items(): # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(q_dot_ix) in sliding_window_instance.state_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_dot_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_mf_dot[int(q_dot_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            q_mf_dot[int(q_dot_ix)-1] = q_dot_val

    return q_mf, q_mf_dot, u_mf


def construct_local_vectors(sliding_window_instance):
    '''helper function to get  q_s, q_s_dot, u_s by reading data from sensors (i.e. blackboard in this case).
     cannot give us p_mf or p_l becuase p is non-physical and must be computed.
    output:  
        q_s, q_s_dot, u_s for local agent
    '''
    bb = sliding_window_instance.bb
    qpu_vec = sliding_window_instance.qpu_vec
    q_s_shape = len(sliding_window_instance.state_indices)
    u_s_shape = len(sliding_window_instance.control_indices)
    q_s = np.zeros(q_s_shape)  # this should be an array of values
    q_s_dot = np.zeros(q_s_shape)  # this should be an array of values
    u_s = np.zeros(u_s_shape) # this should be an array of values
    # get the indices for this agent
    for q_ix, q_val in bb.q_p_u_dict['q_s'].items():
        # if this index does NOT PERTAIN to this agent, then pass 
        # if the index does PERTAIN to this agent, then fill in with value from blackboard
        if int(q_ix) in sliding_window_instance.state_indices:
            # fill in with value from blackboard
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_s[int(q_ix)-1] = qpu_vec[qpu_ix[0][0]]

    for u_ix, u_val in bb.q_p_u_dict['u_s'].items():
        if int(u_ix) in sliding_window_instance.control_indices:
            qpu_ix = np.where(np.array(sliding_window_instance.control_indices)==int(u_ix))
            u_s[int(u_ix)-1] = qpu_vec[qpu_ix[0][0]]

    for q_dot_ix, q_dot_val in bb.q_p_u_dict['q_s_dot'].items(): # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        if int(q_dot_ix) in sliding_window_instance.state_indices:
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_dot_ix))
            q_s_dot[int(q_dot_ix)-1] = qpu_vec[qpu_ix[0][0]]

    return q_s, q_s_dot, u_s


def construct_mf_vectors(sliding_window_instance):
    '''helper function to get  q_mf, q_mf_dot, u_mf (incorrectly called q_s, q_s_dot, u_s, in the q_p_u_dict)
    This method supplements the q_s vector inside of sliding_window_instance with the values from the blackboard so that q_mf contains values for all of the states, not just the local ones.
    '''
    # need to get the values for all states and controls in order to construct q_mf, q_mf_dot, and u_mf
    # get them from the blackboard
    # construct q_mf, q_mf_dot, and u_mf using values from the blackboard
    bb = sliding_window_instance.bb
    qpu_vec = sliding_window_instance.qpu_vec
    q_mf_shape = len(bb.q_p_u_dict['q_s'].items())
    u_mf_shape = len(bb.q_p_u_dict['u_s'].items())
    q_mf = np.zeros(q_mf_shape)  # this should be an array of values
    q_mf_dot = np.zeros(q_mf_shape)  # this should be an array of values
    u_mf = np.zeros(u_mf_shape) # this should be an array of values
    '''
    Get entire state vector from the blackboard, and then overwrite values with local values for the states that pertain to this agent
    '''
    # get the indices for ALL of the states in entire system, from blackboard
    for q_ix, q_val in bb.q_p_u_dict['q_s'].items():
        # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(q_ix) in sliding_window_instance.state_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_mf[int(q_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            q_mf[int(q_ix)-1] = q_val

    for u_ix, u_val in bb.q_p_u_dict['u_s'].items():
        # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(u_ix) in sliding_window_instance.control_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.control_indices)==int(u_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            u_mf[int(u_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            u_mf[int(u_ix)-1] = u_val

    for q_dot_ix, q_dot_val in bb.q_p_u_dict['q_s_dot'].items(): # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from qpu_vec
        if int(q_dot_ix) in sliding_window_instance.state_indices:
            # fill in q_mf with value from qpu_vec
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_dot_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_mf_dot[int(q_dot_ix)-1] = qpu_vec[qpu_ix[0][0]]
        else:
            # fill in q_mf with value from blackboard
            q_mf_dot[int(q_dot_ix)-1] = q_dot_val

    return q_mf, q_mf_dot, u_mf


def construct_local_vectors(sliding_window_instance):
    '''helper function to get  q_s, q_s_dot, u_s by reading data from sensors (i.e. blackboard in this case).
     cannot give us p_mf or p_l becuase p is non-physical and must be computed.
    output:  
        q_s, q_s_dot, u_s for local agent
    '''
    bb = sliding_window_instance.bb
    qpu_vec = sliding_window_instance.qpu_vec
    q_s_shape = len(sliding_window_instance.state_indices)
    u_s_shape = len(sliding_window_instance.control_indices)
    q_s = np.zeros(q_s_shape)  # this should be an array of values
    q_s_dot = np.zeros(q_s_shape)  # this should be an array of values
    u_s = np.zeros(u_s_shape) # this should be an array of values
    # get the indices for this agent
    for q_ix, q_val in bb.q_p_u_dict['q_s'].items():
        # if this index does NOT PERTAIN to this agent, then pass 
        # if the index does PERTAIN to this agent, then fill in with value from blackboard
        if int(q_ix) in sliding_window_instance.state_indices:
            # fill in with value from blackboard
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_s[int(q_ix)-1] = qpu_vec[qpu_ix[0][0]]

    for u_ix, u_val in bb.q_p_u_dict['u_s'].items():
        if int(u_ix) in sliding_window_instance.control_indices:
            qpu_ix = np.where(np.array(sliding_window_instance.control_indices)==int(u_ix))
            u_s[int(u_ix)-1] = qpu_vec[qpu_ix[0][0]]

    for q_dot_ix, q_dot_val in bb.q_p_u_dict['q_s_dot'].items(): # if this index does NOT PERTAIN to this agent, then fill in q_mf with value from the blackboard
        if int(q_dot_ix) in sliding_window_instance.state_indices:
            qpu_ix = np.where(np.array(sliding_window_instance.state_indices)==int(q_dot_ix))
            q_s_dot[int(q_dot_ix)-1] = qpu_vec[qpu_ix[0][0]]

    return q_s, q_s_dot, u_s

