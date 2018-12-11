#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:09:32 2018
Implement Control Law Propagator

"""
import numpy as np 
import abc
import scipy as sp
import ode

from blackboard import *


class SlidingWindow(object):
    '''
    This will be an abstract base class that mandates the user to define the following:
   
     Functions for dynamics, which take in three numpy arrays as inputs, and returns an numpy array
        qp_rhs(q,p,u) 
        u_rhs(q,p,u)
    
    Initial conditions for q,p, and u, (all numpy arrays):
        q_0, p_0, u_0 

    t_0 (start time in float)
    T (duration of time of window in float)
    K (number of intervals for half the window)
    Gamma (algorithmic parameter for Riemann descent)
    t_terminal (end time of control law propagator)
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def qp_rhs(self, q,p,u):
        return

    @abc.abstractmethod
    def u_rhs(self, q,p,u):
        return

    @abc.abstractproperty
    def integrateTol(self):
        return

    @abc.abstractproperty
    def integrateMaxIter(self):
        return

    @abc.abstractproperty
    def q_0(self):
        return

    @abc.abstractproperty
    def p_0(self):
        return

    @abc.abstractproperty
    def u_0(self):
        return

    @abc.abstractproperty
    def t_0(self):
        return

    @abc.abstractproperty
    def T(self):
        return

    @abc.abstractproperty
    def K(self): 
        return

    @abc.abstractproperty
    def Gamma(self): 
        return

    @abc.abstractproperty
    def t_terminal(self): 
        return


class SlidingWindowExample(SlidingWindow):
    '''
    Implementation of SlidingWindow abc
    '''
    def qp_rhs(self, t, qp_vec, **kwargs):
        dim = len(qp_vec)/4
        q = qp_vec[:dim]
	p = qp_vec[dim:2*dim]
        u = kwargs['u_0']
        # for q-dot
        q_dot = np.zeros(np.shape(p))
        # for p-dot
        p_dot = np.zeros(np.shape(q))
        return np.concatenate([q_dot, p_dot])
      
    def u_rhs(self, t, u_vec, **kwargs):
        qp_vec = kwargs['qp_vec']
        dim = len(qp_vec)/4
        q = qp_vec[:dim]
	p = qp_vec[dim:2*dim]
        Gamma = kwargs['Gamma']
        # for u-dot
        return -1*Gamma*np.zeros(np.shape(u_vec))

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
    t_terminal = 2
    n_s = 10

def propagate_dynamics(sliding_window_instance): 
    '''
    Inputs:
        sliding_window_instance (instance of user-defined class which inherits SlidingWindow): object defining the dynamics, and the initial conditions and parameters for numerical integration/propagation. 
    Outputs:
        q_bars, p_bars, u_bars (each is a list of np.arrays): implemented state/costate/control values for entire propagator.
    '''
    q_ls=[]
    p_ls=[]
    p_mfs=[]
    us=[]
    
    q_ls_dot=[]
    p_ls_dot=[]
    p_mfs_dot=[]
    us_dot=[]

    t_0, T, K, integrateTol, integrateMaxIter, state_dim, Gamma = sliding_window_instance.t_0, sliding_window_instance.T, sliding_window_instance.K, sliding_window_instance.integrateTol, sliding_window_instance.integrateMaxIter, sliding_window_instance.state_dim, sliding_window_instance.Gamma 
    weights, weights_total = get_weights(K)
    ts = np.linspace(t_0, T, (2*K)+1)
    qpu_vec = sliding_window_instance.qpu_vec # this qpu_vec is local
    q_s = qpu_vec[:state_dim]
    p_l = qpu_vec[state_dim:2*state_dim]
    H_l_D = sliding_window_instance.H_l_D(q_s, p_l)
    for i in range(len(ts)-1):
        t_start, t_end = ts[i], ts[i+1]
        u_0 = qpu_vec[3*state_dim:]
        # retrieve values from blackboard to pass in as kwargs to the rhs functions inside of propagate_q_p and propagate_u
        q_mf, q_mf_dot, u_mf = construct_mf_vectors(sliding_window_instance) 
        qp_vecs, qp_dot_vecs = propagate_q_p(qpu_vec, t_start, t_end, sliding_window_instance, q_mf, u_mf)  # assume "u" constant, and propagate q and p

        # also need to return derivatives
        # use qp_dot_vecs at the end of each bucket to get the derivatives
        # t=0.0 doesn't matter what the value is here because derivative is not a function of time anyway (it's time invariant)
        qp_dot_vec = qp_dot_vecs[-1]
        q_s_dot = qp_dot_vec[:state_dim]
        p_l_dot = qp_dot_vec[state_dim:2*state_dim]
        p_mf_dot = qp_dot_vec[2*state_dim:]

        # prepend initial condition for q and p for propagating u
        lhs_qp_vecs = [qpu_vec[:-1]] + qp_vecs[:-1] # last item in qpu_vec is "u", so leave it out. last item in qp_vecs is the last point in propagation (since we are using left hand side of q and p - leave it out.
        u_vecs = propagate_u(u_0, lhs_qp_vecs, t_start, t_end, sliding_window_instance, q_s_dot, p_l_dot, p_mf_dot, q_mf_dot, q_mf, u_mf, H_l_D)      # pass in the resulting lhs q and p values to be used for propagating the "u"
        # again t=0.0 doesn't matter what the value is here because derivative is not a function of time anyway (it's time invariant)
        u_dot_vec = u_vecs[-1]

        qpu_vec_i = np.hstack([qp_vecs, u_vecs])
        qpu_vec = qpu_vec_i[-1] # only need the last value
        # Since control, u has changed, the manifold has changed and we must update p_MF and p_l using the same q and q-dot values
        p_mf, p_l = compute_p_mf_p_l(qpu_vec, sliding_window_instance)
        q_s = qpu_vec[:state_dim]
        p_l = qpu_vec[state_dim:2*state_dim]
        p_mf = qpu_vec[2*state_dim:3*state_dim]
        u_s = qpu_vec[3*state_dim:]
 
        if i == len(ts)-2:
            pass
            # no need to append since weight = 0 for last value.  But qpu_vec still needs to be updated.
        else:
            q_ls.append(qpu_vec[:state_dim])
            p_ls.append(qpu_vec[state_dim:2*state_dim])
            p_mfs.append(qpu_vec[2*state_dim:3*state_dim])
            us.append(qpu_vec[3*state_dim:])
            
            q_ls_dot.append(qp_dot_vec[:state_dim])
            p_ls_dot.append(qp_dot_vec[state_dim:2*state_dim])
            p_mfs_dot.append(qp_dot_vec[2*state_dim:])
            us_dot.append(u_dot_vec)
                                 
    q_ls_bar = apply_filter(q_ls, weights, weights_total)
    p_ls_bar = apply_filter(p_ls, weights, weights_total)
    p_mfs_bar = apply_filter(p_mfs, weights, weights_total)
    u_bar = apply_filter(us, weights, weights_total)

    q_ls_dot_bar = apply_filter(q_ls_dot, weights, weights_total)
    p_ls_dot_bar = apply_filter(p_ls_dot, weights, weights_total)
    p_mfs_dot_bar = apply_filter(p_mfs_dot, weights, weights_total)
    u_dot_bar = apply_filter(us_dot, weights, weights_total)
   
    # return the window in time which we will implement these values
    window = (t_0 + (T/2), t_0 + 3*(T/2))
    # need to update values of the qpu_vec inside sliding_window_instance
    return qpu_vec, q_ls_bar, p_ls_bar, p_mfs_bar, u_bar, q_ls, p_ls, p_mfs, us, q_ls_dot_bar, p_ls_dot_bar, p_mfs_dot_bar, p_mfs_dot_bar, u_dot_bar, window  # return values for one entire window


def propagate_q_p(qpu_vec, t_start, t_end, sliding_window_instance, q_mf, u_mf):
    '''
    Propagate q and p to end of bucket
    Inputs:
        qpu_vec (1D numpy array): local qpu_vec containing q_s, p_l, p_mf, u_s, concatenated in one array
        t_start 
        t_end
        sliding_window_instance
        q_mf
        u_mf
    Outputs:
        qp_vecs (list of 1-D numpy arrays): holds qp values for each time interval
        qp_dot_vecs (list of 1-D numpy arrays): holds q_s_dot, p_mf_dot, p_l_dot  values for each time interval
    '''
    state_dim = sliding_window_instance.state_dim
    state_indices = sliding_window_instance.state_indices
    n_s = sliding_window_instance.n_s
    q_l_0 = qpu_vec[:state_dim]
    p_l_0 = qpu_vec[state_dim:2*state_dim]
    p_mf_0 = qpu_vec[2*state_dim:3*state_dim]
    u_0 = qpu_vec[3*state_dim:]
    qp_vecs = []
    qp_vec = np.concatenate([q_l_0, p_l_0, p_mf_0])  # pass in all three: q_0, p_0, u_0, but in the qp_rhs function
    qp_dot_vecs = []
    steps = np.linspace(t_start, t_end, n_s+1)
    # pass in values from blackboard as kwargs to qp_rhs
    for i in range(n_s):
        n_start, n_end = steps[i], steps[i+1]
       
        # update q_mf with the most recent local values in q_s
        q_s = qp_vec[:state_dim]
        q_mf = update_q_mf(q_mf, q_s, sliding_window_instance)
        qp_vec, t, failFlag, iter_i = ode.ode_rk23(sliding_window_instance.qp_rhs, n_start, n_end, qp_vec, sliding_window_instance.integrateTol, sliding_window_instance.integrateMaxIter, state_dim=sliding_window_instance.state_dim, Gamma = sliding_window_instance.Gamma, u_0 = u_0, q_mf=q_mf, u_mf=u_mf)
        qp_vec_orig=qp_vec
        
        # rk23 returns 2 arrays but we remove the first array by doing qp_vec[1] because rk_23 returns the initial value you passed in
        if len(np.shape(qp_vec_orig)) >=2:
            qp_vec = qp_vec[-1]
        qp_vecs.append(qp_vec)
        # get time derivatives
        qp_dot_vec = sliding_window_instance.qp_rhs(0.0, qp_vec, state_dim=sliding_window_instance.state_dim, Gamma = sliding_window_instance.Gamma, u_0 = u_0, q_mf=q_mf, u_mf=u_mf)
        qp_dot_vecs.append(qp_dot_vec)

    return qp_vecs, qp_dot_vecs


def propagate_u(u_0, qp_vecs, t_start, t_end, sliding_window_instance, q_s_dot, p_l_dot, p_mf_dot, q_mf_dot, q_mf, u_mf, H_l_D):
    '''
    Propagate u based on q and p values
    u_vecs (list of 1-D numpy arrays):
    '''
    u_vecs = []
    u_vec = u_0
    n_s = sliding_window_instance.n_s
    steps = np.linspace(t_start,t_end, n_s+1)
    for i in range(n_s):
        n_start, n_end = steps[i], steps[i+1]
        qp_vec = qp_vecs[i]

        u_vec, t, failFlag, iter_i = ode.ode_rk23(sliding_window_instance.u_rhs, n_start, n_end, u_vec, sliding_window_instance.integrateTol, sliding_window_instance.integrateMaxIter, state_dim=sliding_window_instance.state_dim, Gamma = sliding_window_instance.Gamma, qp_vec = qp_vec, u_0=u_0, q_s_dot=q_s_dot, p_l_dot=p_l_dot, p_mf_dot=p_mf_dot, q_mf_dot=q_mf_dot, q_mf=q_mf, u_mf=u_mf, H_l_D=H_l_D)
        if len(u_vec)>1:
            u_vec_next=u_vec[-1]
        else:
            u_vec_next=u_vec
        u_vecs.append(u_vec_next) # one u_vec for each step, append them and you have all the u_vecs for one bucket
        u_vec=u_vec_next
    return u_vecs


def get_weights(K):
    '''
    Inputs:
        K (int): number of values for half of the sliding window
    Outputs:
        weights (float): weights to be used for weighting values in the window.
        weights_total (float): sum of all of the weights for entire window
    ''' 
    weights_0 = [float(i)/K for i in range(1,K+1)]  
    weights_1 = [2-(float(i)/K) for i in range(K+1,(2*K)+1)]
    # sanity check 
    assert len(weights_0)==len(weights_1)
    weights = weights_0+weights_1
    weights_total = sum(weights[:-1])
    return weights, weights_total 

def apply_filter(vec, weights, weights_total):
    '''
    Inputs:
        vec (np.array): vector of state/control values to apply window filter to
        weights (list of floats): weights for middle values of window.  End points are excluded because weights are 0 at endpoints.
        weights_total (float): sum of all weights for entire window.
    Outputs:
        vec_normalized (np.array): normalized vector of state/control values
    ''' 

    vec_weighted = [val*w for val,w in zip(vec, weights[:-1])]
    vec_current = np.sum(vec_weighted,0)
    vec_normalized = vec_current/float(weights_total)
    return vec_normalized

def compute_p_mf_p_l(qpu_vec, sliding_window_instance):
    state_dim = sliding_window_instance.state_dim
    # need to prepare the q_mf, q_mf_dot, and u_mf vectors
    q_mf, q_mf_dot, u_mf = construct_mf_vectors(sliding_window_instance)
    q_s = qpu_vec[:state_dim]
    u_s = qpu_vec[3*state_dim:]
    q_s_dot = sliding_window_instance.q_s_dot
    # use the most up-to-date values to compute p_l and p_mf
    p_l = sliding_window_instance.L_l_q_dot(q_s, q_s_dot, u_s)
    p_mf = sliding_window_instance.L_mf_q_dot(q_mf, q_mf_dot, u_mf)
    return p_mf, p_l



def update_q_mf(q_mf, q_s, sliding_window_instance):
    '''
    helper function that updates the mean field vectors with the local state, control computed during propagation
    Output:
        q_mf (np array): q_mf, with the local values overwritten by the most recent values from q_s
        (p_mf gets updated automatically, and u_s is constant, so u_mf also stays the same)
    The procedure for propagating q and p will be:
        - update q_mf with the q_s just computed from propagation
        - (p_mf gets updated automatically, and u_s is constant, so u_mf also stays the same)
        - call ode_rk23 on qp_rhs to propagate q and p
        - call qp_rhs *directly* in order to get q_s_dot, p_mf_dot, and p_l_dot
        - record all of the q_s, p_mf, and p_l in qp_vecs
        - record all of the q_s_dot, p_mf_dot, and p_l_dot in qp_dot_vecs
        return qp_vecs, and qp_dot_vecs at the very end, and then 
        inside of propagate_dynamics, instead of computing  
    '''
    for ix, q_val in enumerate(q_mf):
        # if this index does NOT PERTAIN to this agent, then pass
        # if the index does PERTAIN to this agent, then fill in q_mf with the value from q_s
        q_ix = ix+1
        if int(q_ix) in sliding_window_instance.state_indices:
            # fill in q_mf with value from q_s
            q_s_ix  = np.where(np.array(sliding_window_instance.state_indices)==int(q_ix))
            # TODO: write an assertion to make sure qpu_ix has exactly 1 element (not 0, and not more than 1)
            q_mf[ix] = q_s[q_s_ix[0][0]]
        else:
            pass

    return q_mf


def sliding_window(sliding_window_instance):
    ''' 
    This method runs the propagation for a single agent.  Corresponding to the flow chart it runs:
        - Read from blackboard to get the following observation measured at time t_0, and onwards
        - construct quenched mean field for Hamiltonian agent i
        - Setup initial conditions for L_MF and p_MF
        - Construct agent synchronized Hamiltonian and partial derivatives
    
    Inputs:
    The only input is sliding_window_instance, but we use the following attributes of the sliding_window_instance:
        t_0 (int): Initial time to start propagating dynamics
        T (int): End time of propagating dynamics
        q_0 (np.array): initial values of state vector
        p_0 (np.array): initial values of costate vector
        u_0 (np.array): initial values of control vector
        state_dim (int): number of states
        Gamma (float): algorithmic parameter for Riemann descent algorithm
        t_terminal (int): time marking termination of control law propagator algorithm
    Outputs:
        q_bars, p_bars, u_bars (list of np.arrays): implemented state/costate/control values for entire propagator.
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
        print t
        # for the times, propagate_dynamics needs: t_0, T, and K.  T and K can come from the sliding_window_instance
        #...t_0 will be passed in.  t_0 is the start of the window.

        # this propagates a single window
        # inside of propagate dynamics
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

