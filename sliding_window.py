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


class SlidingWindow(object):
    '''
    This will be an abstract base class that mandates the user to define the following:
   
     Functions for dynamics, which take in three numpy arrays as inputs, and returns an numpy array
        qp_rhs(q,p,u) 
        u_rhs(q,p,u)
    
    Initial conditions for q,p, and u, (all numpy arrays):
        q_0, p_0, u_0 

    t_0 (start time in float)
    T (end time of window in float)
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
        q_D = qp_vec[2*dim:3*dim]
        p_D = qp_vec[3*dim:]
        u = kwargs['u_0']
        # for q-dot
        q_dot = np.zeros(np.shape(p))
        # for p-dot
        p_dot = np.zeros(np.shape(q))
        q_D_dot = np.zeros(np.shape(p))
        p_D_dot = np.zeros(np.shape(q))
        return np.concatenate([q_dot, p_dot, q_D_dot, p_D_dot])
      
    def u_rhs(self, t, u_vec, **kwargs):
        qp_vec = kwargs['qp_vec']
        dim = len(qp_vec)/4
        q = qp_vec[:dim]
	p = qp_vec[dim:2*dim]
        q_D = qp_vec[2*dim:3*dim]
        p_D = qp_vec[3*dim:]
        Gamma = kwargs['Gamma']
        # for u-dot
        return -1*Gamma*np.zeros(np.shape(u_vec))

    # Inputs for numerical propagator
    q_0 = np.array([0])
    p_0 = np.array([0])
    q_D = np.array([0])
    p_D = np.array([0])
    u_0 = np.array([0])
    qpu_vec = np.hstack([q_0, p_0, q_D, p_D, u_0])
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
    qs=[]
    ps=[]
    q_Ds=[]
    p_Ds=[]
    us=[]
    t_0, T, K, integrateTol, integrateMaxIter, state_dim, Gamma = sliding_window_instance.t_0, sliding_window_instance.T, sliding_window_instance.K, sliding_window_instance.integrateTol, sliding_window_instance.integrateMaxIter, sliding_window_instance.state_dim, sliding_window_instance.Gamma 
    weights, weights_total = get_weights(K)
    ts = np.linspace(t_0, T, (2*K)+1)
    qpu_vec = sliding_window_instance.qpu_vec
    for i in range(len(ts)-1):
        t_start, t_end = ts[i], ts[i+1]
        u_0 = qpu_vec[4*state_dim:]
        qp_vecs = propagate_q_p(qpu_vec, t_start, t_end, sliding_window_instance)  # assume "u" constant, and propagate q and p
        # prepend initial condition for q and p for propagating u
        lhs_qp_vecs = [qpu_vec[-1:]] + qp_vecs[:-1] # last item in qpu_vec is "u", so leave it out. last item in qp_vecs is the last point in propagation (since we are using left hand side of q and p - leave it out.
        u_vecs = propagate_u(u_0, lhs_qp_vecs, t_start, t_end, sliding_window_instance)      # pass in the resulting lhs q and p values to be used for propagating the "u"
        qpu_vec_i = np.hstack([qp_vecs, u_vecs])
        qpu_vec = qpu_vec_i[-1] # only need the last value
        if i == len(ts)-2:
            pass
            # no need to append since weight = 0 for last value.  But qpu_vec still needs to be updated.
        else:
            qs.append(qpu_vec[:state_dim])
            ps.append(qpu_vec[state_dim:2*state_dim])
            q_Ds.append(qpu_vec[2*state_dim:3*state_dim])
            p_Ds.append(qpu_vec[3*state_dim:4*state_dim])
            us.append(qpu_vec[4*state_dim:])
    q_bar = apply_filter(qs, weights, weights_total)
    p_bar = apply_filter(ps, weights, weights_total)
    q_D_bar = apply_filter(q_Ds, weights, weights_total)
    p_D_bar = apply_filter(p_Ds, weights, weights_total)
    u_bar = apply_filter(us, weights, weights_total)
    return qpu_vec, q_bar, p_bar, q_D_bar, p_D_bar,  u_bar, qs, ps, q_Ds, p_Ds, us  # return values for one entire window

    
def propagate_q_p(qpu_vec, t_start, t_end, sliding_window_instance):
    '''
    Propagate q and p to end of bucket using rk23
    '''
    state_dim = sliding_window_instance.state_dim
    n_s = sliding_window_instance.n_s
    q_0 = qpu_vec[:state_dim]
    p_0 = qpu_vec[state_dim:2*state_dim]
    q_D = qpu_vec[2*state_dim:3*state_dim]
    p_D = qpu_vec[3*state_dim:4*state_dim]
    u_0 = qpu_vec[4*state_dim:]
    qp_vecs = []
    qp_vec = np.concatenate([q_0, p_0, q_D, p_D])  # pass in all three: q_0, p_0, u_0, but in the qp_rhs function
    steps = np.linspace(t_start, t_end, n_s+1)
    import pdb; pdb.set_trace()
    for i in range(n_s):
        n_start, n_end = steps[i], steps[i+1]
        qp_vec, t, failFlag, iter_i = ode.ode_rk23(sliding_window_instance.qp_rhs, n_start, n_end, qp_vec, sliding_window_instance.integrateTol, sliding_window_instance.integrateMaxIter, state_dim=sliding_window_instance.state_dim, Gamma = sliding_window_instance.Gamma, u_0 = u_0)
        qp_vecs.append(qp_vec[-1])
        qp_vec = qp_vec[-1]
    return qp_vecs


def propagate_u(u_0, qp_vecs, t_start, t_end, sliding_window_instance):
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
        u_vec, t, failFlag, iter_i = ode.ode_rk23(sliding_window_instance.u_rhs, n_start, n_end, u_vec, sliding_window_instance.integrateTol, sliding_window_instance.integrateMaxIter, state_dim=sliding_window_instance.state_dim, Gamma = sliding_window_instance.Gamma, qp_vec = qp_vec)
        u_vecs.append(u_vec[-1]) # one u_vec for each step, append them and you have all the u_vecs for one bucket
        u_vec = u_vec[-1]
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


