#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:06:19 2018
Cellection of numerical methods for ordinary differential equations


"""

import numpy as np
import math

def ode_rk23(F, a, b, y0, tol, maxIter, **kwargs):
    '''
    %--------------------------------------------------------------------------
    %   Author  : Originally Developed by Richard Wang, Clearsight Systems Inc. 
    %
    %   History : Original developed on 7/25/2006 in MATLAB
    %             Transfered into Python on 4/24/2018
    %  Purpose  : Compute the solution y for x between a and b to a 
    %             specified accuracy tol for the following first order 
    %             initial value problem 
    %
    %                   dy/dx = F(x,y)     
    %             with
    %                   y(a) = y0
    %  
    %             using a pair of Runge-Kutta formulae of orders 2 and 3 
    %             with automatic step length control.
    %   
    %  Input parameters:
    %   F        :  function name for the right-hand side of the differential 
    %               equation F(x,y,...)
    %   a,b      :  the interval for the independent variable; 
    %               type - float
    %   y0       :  the initial value of y at y0; 
    %               type - 1D numpy.ndarray
    %   tol      :  local error tolerance for the solution
    %               type - float > 0
    %   maxIter  :  maximal iterations allowed for executing nemerical integration
    %               type - int > 0
    %   **kwargs :  additional parameters for defining the right-hand
    %               side function F(x,y,...)
    %
    %  Output parameters:
    %   x        :  the values of independent variable where solution is
    %               computed; 
    %               type: 1D numpy.ndarray
    %   y        :  the solution at x; 
    %               type: 2D numpy.ndarray
    %
    %    Remarks:
    %     1) It is user's responsibility to ensure that y0 is an 
    %        1D numpy.ndarray (i.e. n-dimensional vector) and tol > 0 on input 
    %     2) On output, y is the n-dimensional row vector such that y[i] is 
    %        the compute solution at x[i] for i=0,...,n-1 with x(0)=a, x(n-1)=b
    %     3) The routine will exit with an error after the step size has
    %        been reduced to no larger than 1e-6 of the length of b-a 
    %
    % ------------------------------------------------------------------------
    %   Author  : Richard Wang, Clearsight Systems
    %
    %   History : Original developed on 7/25/2006 
    %
    ''' 

    # Initialization
    tTime = np.array([a])
    failFlag = False
    y = np.array(y0)    
    iter_i = np.int(0)
    hStep = b-a
    try:
        if abs(hStep) < 1e-16:
            # To do: send back message: the interval for the integration is almost 0
            pass
        else:
            hmin = (1e-10)*abs(hStep)
            m = len(y0)
            ONE_m = np.ones(m)
            
            t = a
            y1 = y0  # should be row vector
            
            k1 =  F(t, y1, **kwargs) # should be a row vector
            # Procedure for Runge-Kutta 2nd & 3rd orders formulae with
            # automatic step control    
            while not failFlag:
            
                iter_i = iter_i+1
                
                if iter_i == maxIter:
                    failFlag = True
                else:
                    hk1 = hStep*k1 # should be a row vector
                    k2 = F(t+hStep/2.0, y1+hk1/2.0, **kwargs) #feval(F, t+h/2., y1+hk1/2., varargin{1:end});
                    k3 = F(t+0.75*hStep, y1+0.75*hStep*k2, **kwargs) #feval(F, t+0.75*h, y1+0.75*h*k2, varargin{1:end});
                    k40 = y1 + hStep*(2.0*k1+ 3.0*k2+ 4.0*k3)/9.0
                    k4 = F(t+hStep, k40, **kwargs) #feval(F, t+h, k40, varargin{1:end});
                    
                    # Establish the scalar vector for the solution
                    yscale = abs(y1)+abs(hk1)
                    
                    yscale = np.maximum(yscale, ONE_m)
                    
                    # Estimation the error for the potential solution at t
                    R = abs((-5.0*k1+6.0*k2+8.0*k3-9.0*k4)*hStep)/72.0
                    
                    # Acceptance test for the solution at t
                    R1 = max(R/yscale)
                    R2 = R1/tol
                    
                    # Accept the solution at t when its required accuracy is fulfilled
                    if (R2 < 1):
                        # Update t,y,k1 for future computation and store (tTime,y) for output
                        t = t+hStep
                        y1 = k40
                        k1 = k4
                        tTime = np.append(tTime, t)
                        y = np.row_stack((y, y1)) 
                        if R2 != 0:  
                            hStep = hStep* min(1.5, 0.95*math.pow(R2, -1.0/3.0)) # increasing step size
                        else:
                            hStep =  2.0 # arbitrarily chosen to be larger than 2.5, because R = 0 implies zero error
                        #  Check if solution has been done for tTime = b
                        if (t == b):
                            break
                        else:
                            # Determine the new step size when it is actually too large
                            if (hStep > 0.0 and t+hStep > b) or (hStep < 0.0 and t+hStep < b):
                                hStep = b-t                    
                    else:   # reduce step size  due to the failure of accuacy test
                        hStep = hStep* max(0.5, 0.95*math.pow(R2, -1.0/2.0));
                        if (abs(hStep) <= hmin):
                            failFlag=True
    except Exception as e:
        print("ode error: " + str(e))
        failFlag = True
    

    return y, tTime, failFlag, iter_i
     

     
