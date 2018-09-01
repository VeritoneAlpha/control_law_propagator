#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:09:32 2018
Implement Control Law Propagator

"""
import numpy as np
import scipy as sp
import ode
from clp_init import *


def rhs(t, qpu_vec, **kwargs):
    '''
    Inputs:
        t (integer): time at which the differential equations are being evaluated
        qpu_vec (2D np.array): value at current time to be propagated forward according to dynamics.
        **kwargs:  arguments passed to corresponding functions for dynamics of q, p, and u.
    Outputs:
        qpu_dot_vec (2D np.array): values for differential equations for  q, p, and u at current time interval.  These values are consumed by the numerical integration to propagate the dynamics.
    '''
    state_dim = kwargs['state_dim']
    Gamma = kwargs['Gamma']
    q = qpu_vec[:state_dim]
    p = qpu_vec[state_dim:2*state_dim]
    u = qpu_vec[2*state_dim:]
    q_dot =  H_T_p(q,p,u)
    p_dot = -1*H_T_q(q,p,u)
    u_dot = -Gamma*Q_u(q,p,u)
    qpu_dot_vec = np.hstack([q_dot, p_dot, u_dot])
    return qpu_dot_vec

def propagate_dynamics(t_0, T, K, qpu_vec, integrateTol, integrateMaxIter, state_dim, Gamma):
    qs=[]
    ps=[]
    us=[]

    ts = range(t_0,T+1,(T-t_0)/(2*K)) # go until T+1 because last value will be used as starting point for next window

    for i in range(len(ts)-1):
        t_start, t_end = ts[i], ts[i+1]
        qpu_vec_i, t, failFlag, iter_i = ode.ode_rk23(rhs, t_start, t_end, qpu_vec, integrateTol, integrateMaxIter, state_dim=state_dim, Gamma = Gamma)
        qpu_vec = qpu_vec_i[-1] # only need the last value
        if i == len(ts)-2 :
            pass 
            # no need to append since weight = 0 for last value.  but qpu_vec still needs to be updated.
        else:
            qs.append(qpu_vec[:state_dim])
            ps.append(qpu_vec[state_dim:2*state_dim])
            us.append(qpu_vec[2*state_dim:])

    return qpu_vec, qs, ps, us
       
def get_weights(K):
    weights_0 = [float(i)/K for i in range(1,K+1)]  
    weights_1 = [2-(float(i)/K) for i in range(K+1,(2*K)+1)]
    # sanity check 
    assert len(weights_0)==len(weights_1)
    weights = weights_0+weights_1
    weights_total = sum(weights[:-1])
    return weights, weights_total

def apply_filter(vec, weights, weights_total):
    vec_weighted = [val*w for val,w in zip(vec, weights[:-1])]
    vec_current = np.sum(vec_weighted,0)
    vec_normalized = vec_current/float(weights_total)
    return vec_normalized

def sliding_window(t_0, T, K, q_0, p_0, u_0, state_dim, Gamma, t_terminal):
    q_bars = []
    p_bars = []
    u_bars = []
    weights, weights_total = get_weights(K)
    t=t_0 # wall clock time
    qpu_vec = np.hstack([q_0, p_0, u_0])
    while t<t_terminal:
        
        qpu_vec, qs, ps, us = propagate_dynamics(t_0, T, K, qpu_vec, integrateTol, integrateMaxIter, state_dim, Gamma)
        # qs, ps, and us will go to Mean Field somehow

        q_bar = apply_filter(qs,weights, weights_total)
        p_bar = apply_filter(ps,weights, weights_total)
        u_bar = apply_filter(us,weights, weights_total)
        
        t+=1
        
        q_bars.append(q_bar)
        p_bars.append(p_bar)
        u_bars.append(u_bar)

    return q_bars, p_bars, u_bars

q_bars, p_bars, u_bars = sliding_window(t_0, T, K, q_0, p_0, u_0, state_dim, Gamma, t_terminal)
print 'sliding window finished propagating. Resulting values are:'
print q_bars
print p_bars
print u_bars
