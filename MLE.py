# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:17:02 2022

@author: damia
"""

import numpy as np
import math as ms
from scipy.stats import norm
import pandas as pd
from scipy.optimize import minimize 


def Kalman_repar(y, q): # see section 2.10.2
    n = len(y)
    
    # create empty arrays 
    a        = np.zeros((n,1,1))
    a[0] = np.array([[y[0]]])
    P_t_star = np.zeros((n,1,1))
    P_t_star[0] = np.array([[1 + q]])
    v_t      = np.zeros((n-1,1,1))
    F_t_star = np.zeros((n-1,1,1))
    K_t      = np.zeros((n-1,1,1))
    
    for i in range(n - 1):
        v_t[i]          = y[i] - a[i]
        F_t_star[i]     = P_t_star[i] + np.array([[1]])
        K_t[i]          = P_t_star[i] / F_t_star[i]
        a[i + 1]        = a[i] + np.dot(K_t[i], v_t[i])
        P_t_star[i + 1] = np.dot(P_t_star[i],(np.array([[1]]) - K_t[i])) + np.array([[q]])

    return v_t, F_t_star





data  = pd.read_excel("Nile.xlsx", names =["year", "volume"])
y     = np.array(data["volume"])





def Log_LDC(psi): # eq 2.63
    q = ms.exp(psi)
    v_t, F_t_star  = Kalman_repar(y, q)
    n = len(v_t)
    sigma2_eps_hat = (1 / (n - 1)) * np.sum((v_t ** 2) / F_t_star)
    value = ((- n) / 2) * ms.log(2 * ms.pi) - (n - 1) / 2 - ((n - 1) / 2) * ms.log(sigma2_eps_hat) - (1 / 2)* np.sum(np.log(F_t_star))

    return -1 * value 


result = minimize(Log_LDC, 0, method = 'BFGS')