
# coding: utf-8

# This is the prototype of the Mean Field
# 
# There will be three main modules:
# 
# - 1) Agent
# 
# - 2) MeanField
# 
# - 3) BlackBoard

# In[1]:

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:

import numpy as np
import scipy as sp
import ode
from numerical_propagator import *
from synchronizer import *
from blackboard import *
from batteryAgent import *


# ### batteryAgent

# states: 
# 
# state 1: $ q_1 $
# 
# state 2: $ q_B $
# 
# controls:
# 
# $u_B$
# 

# In[3]:

bb=Blackboard()


# In[4]:

bat = batteryAgent(bb, state_indices=[1,2], control_indices=[1])


# In[5]:

bat.L_l(1,233,3,4,5,3,4,5,3,3,4,5,3,2,3,4)


# In[6]:

bat.H_l(1,233,3,43,5,3,49,59,93,93,4,5,3,21,2,3)


# Test q_rhs_H_l_nou, and p_rhs_H_l_nou

# In[7]:

np.concatenate([np.array([9]),np.array([10])])


# In[8]:

bat.q_rhs_H_l_nou(1,233,3,43,5,3,49,59,93,93,4,5,3,21,2)


# In[9]:

bat.p_rhs_H_l_nou(1,233,3,43,5,3,49,59,93,93,4,5,3,21,2)


# In[10]:

bat.q_rhs_H_l_u(1,233,3,43,5,3,49,59,93,93,4,5,3,21,2)


# In[11]:

bat.p_rhs_H_l_u(1,233,3,43,5,3,49,59,93,93,4,5,3,21,2)


# In[12]:

bat.q_rhs_H_l(1,233,3,43,5,3,49,59,93,93,4,5,3,21,2,8)


# In[13]:

bat.p_rhs_H_l(1,233,3,43,5,3,49,59,93,93,4,5,3,21,2,8)


# In[14]:

bat.qp_rhs_H_l(1,233,3,43,5,3,49,59,93,93,4,5,3,21,2,3)


# In[15]:

bat.q_rhs_H_mf_u(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4)


# In[16]:

bat.p_rhs_H_mf_u(3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4,1)


# In[17]:

bat.q_rhs_H_mf(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4,1)


# In[18]:

bat.p_rhs_H_l(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4,1)


# In[19]:

bat.p_rhs_H_l_u(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4)


# In[20]:

bat.q_rhs_H_l_u(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4)


# In[21]:

bat.p_rhs_H_l_nou(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4)


# In[22]:

bat.qp_rhs(2)


# In[ ]:

bat.q_rhs_H_l(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4,3)


# In[ ]:

bat.q_rhs_H_mf(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4,3)


# In[ ]:

bat.qp_rhs(2,3,4,1, 2,3,4,1, 2,3,4,1, 2,3,4,3)


# ### write u_rhs and test with numerical propagator

# In[25]:

# What things need to call qp_rhs?
'''
Need to fit everything into these inputs:
qp_vec, state_dim=sliding_window_instance.state_dim, Gamma = sliding_window_instance.Gamma, u_0 = u_0, q_mf=q_mf, u_mf=u_mf
'''


# In[26]:

# propagate_q_p(qpu_vec, t_start, t_end, sliding_window_instance, q_mf, u_mf):


# In[27]:

# for right now, doesn't matter what we fill in for the values of the states of the other agents, so for now make =0
u_0 = [u_B]
q_mf = [q_B, q_1, q_1_a, q_1_f, q_1_w, phi_L_1, q_2_a, q_2_f, q_2_w, phi_2_L]
u_mf = [u_B, u_1_1, u_1_2, u_2_1]


# In[29]:

'''remember that the propagator must stay generalized!  cannot hardcode that'''


# In[32]:

't_0' in bat.__dict__


# In[33]:

'''
Make a complete batteryAgent, and then pass it into the sliding_window function.  See where it breaks
'''


# In[36]:

bat.t_0


# In[40]:

bat.q_s_dot


# In[42]:

bat.q_s_0


# In[41]:

sliding_window(bat)


# In[ ]:



