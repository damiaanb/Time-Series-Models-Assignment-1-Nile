import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import norm
import math as ms 
import numpy as np
from scipy.optimize import minimize 
import seaborn as sns 
import statsmodels.api as sm

#%% Figure 2.5 specific Functions


def removedata(y, first, last):
    """
    Converts y to an array with and for the entry number first to last a value of NaN
    """
    y = np.array(y).astype(float)
    y[first:last] = np.nan 
    
    return y
    
#%% Figure 2.6 specific Functions

def empty_forecast(x, y, n_for): 
    """
    Returns x and y including empty n_for entries
    x and y are np.arrays
    n_for is an integer for the number of forecasts 
    """
    n = len(x)
    x_res = np.zeros(n + n_for)
    x_res[0:n] = np.array(x)
    y_res = np.zeros(n + n_for)
    y_res[0:n] = np.array(y)
    x_res[n:(n + n_for)] = np.arange(x[n-1] + 1, x[n-1] + n_for + 1, 1)
    y_res[n:(n + n_for)] = np.nan
 
    return x_res, y_res


def Kalman_Filter_Forecast(y, a1, p1, sigma2_eps, sigma2_eta):
    """
    2 Differences with Kalman_Filter:
        1. For missing observations it sets  F_t[i]  = np.dot(np.dot(Z_t, P[i]), Z_t.T) + H_t in stead of F_t[i] = np.array([[10 ** 7]])
        2. It produces a 50% confidence interval only for the missing values (in this the forecasts) 
           instead of a 95% confidence interval for the observed values 
         
    """
    n = len(y)
    
    # create empty arrays 
    a = np.zeros((n + 1,1,1))
    a[0] = np.array([[a1]])
    P = np.zeros((n + 1,1,1))
    P[0] = np.array([[p1]])
    v_t = np.zeros((n,1,1))
    F_t = np.zeros((n,1,1))
    K_t = np.zeros((n,1,1))
    L_t = np.zeros((n,1,1))
    q1  = np.zeros((n,1,1))
    q2  = np.zeros((n,1,1))
    
    # System matrices of the general state space form 
    Z_t = np.array([[1]])
    H_t = np.array([[sigma2_eps]])
    T_t = np.array([[1]])
    R_t = np.array([[1]])
    Q_t = np.array([[sigma2_eta]])
    
    for i in range(n):
        
        if (np.isnan(y[i])):
            
            v_t[i]     = np.array([[ms.nan]])
            F_t[i]     = np.dot(np.dot(Z_t, P[i]), Z_t.T) + H_t
            K_t[i]     = np.array([[0]])
            L_t[i]     = T_t - np.dot(K_t[i], Z_t)
            q1[i]      = a[i] - 0.5 * ms.sqrt(P[i])
            q2[i]      = a[i] + 0.5 * ms.sqrt(P[i])
            a[i + 1]   = np.dot(T_t, a[i]) 
            P[i + 1]   = np.dot(np.dot(T_t, P[i]), T_t.T) + np.dot(np.dot(R_t, Q_t), R_t.T) 
            
        else:
            
            v_t[i]  = y[i] - np.dot(Z_t, a[i])
            F_t[i]  = np.dot(np.dot(Z_t, P[i]), Z_t.T) + H_t
            K_t[i]  = np.dot(np.dot(np.dot(T_t, P[i]), Z_t.T), inverse(F_t[i]))
            L_t[i]  = T_t - np.dot(K_t[i], Z_t)
            a[i + 1]   = np.dot(T_t, a[i]) + np.dot(K_t[i], v_t[i])
            P[i + 1]   = np.dot(np.dot(T_t, P[i]), T_t.T) + np.dot(np.dot(R_t, Q_t), R_t.T) - np.dot(np.dot(K_t[i], F_t[i]), K_t[i].T)

    return a, P, v_t, F_t, K_t, L_t, q1, q2, n

#%% Figure 2.7 and 2.8 specific Functions

def Kalman_repar(q): # see section 2.10.2
    """
    Reparameterized Kalman filter as in sectionm (2.10.2) of the book of SJ Koopman
    """
    data  = pd.read_csv ('annual-sea-surface-temp-anomaly.csv')
    data = data.loc[data['Entity'] == 'World']
    y     = np.array(data["annual_sea_surface_temperature_anomaly"])
    n = len(y)
    T_t     = np.array([[1]])
    Z_t     = np.array([[1]])
    # create empty arrays 
    a        = np.zeros((n,1,1))
    a[0] = np.array([[y[0]]])
    P_t_star = np.zeros((n,1,1))
    P_t_star[0] = np.array([[1 + q]])
    v_t      = np.zeros((n-1,1,1))
    F_t_star = np.zeros((n-1,1,1))
    K_t      = np.zeros((n-1,1,1))
    L_t      = np.zeros((n-1,1,1))
    
    for i in range(n - 1):
        v_t[i]          = y[i+1] - a[i]
        F_t_star[i]     = P_t_star[i] + np.array([[1]])
        K_t[i]          = P_t_star[i] / F_t_star[i]
        L_t[i]          = T_t - np.dot(K_t[i], Z_t)
        a[i + 1]        = a[i] + np.dot(K_t[i], v_t[i])
        P_t_star[i + 1] = np.dot(P_t_star[i],(np.array([[1]]) - K_t[i])) + np.array([[q]])

    return v_t, F_t_star, L_t, a, P_t_star, y, K_t


def compute_sigma_hat(n, v_t, F_t_star):
    """
    estimator for sigma_epsilon^2 given in equation (2.62) of the book by SJ Koopman
    """
    return (1 / (n - 1)) * np.sum((v_t ** 2) / F_t_star)


def Log_LDC(psi): # eq 2.63
    """
    concentrated diffuse loglikelihood in equation (2.63) of the book by SJ Koopman
    """
    q = ms.exp(psi)
    v_t, F_t_star, L_t, a, P_t_star, y, K_t  = Kalman_repar(q)
    n = len(v_t) + 1
    sigma2_eps_hat = compute_sigma_hat(n, v_t, F_t_star)
    value =  - ((n - 1) / 2) * ms.log(sigma2_eps_hat) - (1 / 2)* np.sum(np.log(F_t_star))

    return -1 * value 


def get_MLE():
    """
    Get the MLE of sigma_epsilon^2 and sigma_eta^2
    """
    result                                    = minimize(Log_LDC, 0, method = 'BFGS')
    psi_opt                                   = float(result.x)
    q_opt                                     = ms.exp(psi_opt)
    v_t, F_t_star, L_t, a, P_t_star, y, K_t   = Kalman_repar(q_opt)
    sigma2_eps_opt                            = compute_sigma_hat(len(v_t) + 1, v_t, F_t_star)
    sigma2_eta_opt                            = q_opt * sigma2_eps_opt
    return sigma2_eps_opt, sigma2_eta_opt

# %% GENERAL FUNCTIONS used for several Figures 

def inverse(M):
    """
    Take inverse of a matrix M
    """
    if M.size == 1:
        res = np.array([[1 / M[0][0]]])
    else:
        res = np.linalg.inv(M)
        
    return(res)

def Kalman_Filter(y, a1, p1, sigma2_eps, sigma2_eta):
    """
    Kalman filter for the general state space model
    see slide 31 of lecture slides of week 2 
    """
    n = len(y)
    
    # create empty arrays 
    a = np.zeros((n + 1,1,1))
    a[0] = np.array([[a1]])
    P = np.zeros((n + 1,1,1))
    P[0] = np.array([[p1]])
    v_t = np.zeros((n,1,1))
    F_t = np.zeros((n,1,1))
    K_t = np.zeros((n,1,1))
    L_t = np.zeros((n,1,1))
    q005 = np.zeros((n,1,1))
    q095 = np.zeros((n,1,1))
    
    # System matrices of the general state space form 
    Z_t = np.array([[1]])
    H_t = np.array([[sigma2_eps]])
    T_t = np.array([[1]])
    R_t = np.array([[1]])
    Q_t = np.array([[sigma2_eta]])
    
    for i in range(n):
        
        if (np.isnan(y[i])):
            
            v_t[i]     = np.array([[ms.nan]])
            F_t[i]     = np.array([[10 ** 7]])
            K_t[i]     = np.array([[0]])
            L_t[i]     = T_t - np.dot(K_t[i], Z_t)
            a[i + 1]   = np.dot(T_t, a[i]) 
            P[i + 1]   = np.dot(np.dot(T_t, P[i]), T_t.T) + np.dot(np.dot(R_t, Q_t), R_t.T) 
            
        else:
            
            v_t[i]  = y[i] - np.dot(Z_t, a[i])
            F_t[i]  = np.dot(np.dot(Z_t, P[i]), Z_t.T) + H_t
            K_t[i]  = np.dot(np.dot(np.dot(T_t, P[i]), Z_t.T), inverse(F_t[i]))
            L_t[i]  = T_t - np.dot(K_t[i], Z_t)
            q005[i] = np.array([[norm.ppf(0.05, loc = float(a[i]), scale = ms.sqrt(float(P[i])))]])
            q095[i] = np.array([[norm.ppf(0.95, loc = float(a[i]), scale = ms.sqrt(float(P[i])))]])
            a[i + 1]   = np.dot(T_t, a[i]) + np.dot(K_t[i], v_t[i])
            P[i + 1]   = np.dot(np.dot(T_t, P[i]), T_t.T) + np.dot(np.dot(R_t, Q_t), R_t.T) - np.dot(np.dot(K_t[i], F_t[i]), K_t[i].T)

    return a, P, v_t, F_t, K_t, L_t, q005, q095, n


def  Kalman_Smoother(n, v_t, F_t, L_t, a, P, y, K_t, sigma2_eps, sigma2_eta):
      """
      Input are the sequences produced during the Kalman Filter process,
      but returns the backward recursion Kalman Smoothing results 
      """
      Z_t = np.array([[1]]) 
      H_t = np.array([[sigma2_eps]])
      Q_t = np.array([[sigma2_eta]])
     
      r_t        = np.zeros((n,1,1))
      r_t[n - 1] = np.array([[0]])
      alpha_hat  = np.zeros((n,1,1))
     
      N_t        = np.zeros((n,1,1))
      N_t[n - 1] = np.array([[0]])
      V_t        = np.zeros((n,1,1))
      V_t[n - 1] = np.array([[0]])
      q005 = np.zeros((n,1,1))
      q095 = np.zeros((n,1,1))
     
      eps_hat    = np.zeros((n,1,1))
      var_eps_yn = np.zeros((n,1,1))
      eta_hat    = np.zeros((n,1,1))
      var_eta_yn = np.zeros((n,1,1))
      D          = np.zeros((n,1,1))
      
      for j in range(n-1,-1,-1):
         
          if (np.isnan(y[j])):
             
              r_t[j - 1]   =  r_t[j]
              alpha_hat[j] = a[j] + np.dot(P[j], r_t[j - 1])
              N_t[j - 1]   = np.dot(np.dot(Z_t.T, inverse(F_t[j])), Z_t) +  N_t[j]
              V_t[j]       = P[j] - np.dot(np.dot(P[j], N_t[j - 1]), P[j])

          else:
            
              r_t[j - 1]   = np.dot(v_t[j], inverse(F_t[j])) + np.dot(L_t[j], r_t[j])
              alpha_hat[j] = a[j] + np.dot(P[j], r_t[j - 1])
              N_t[j - 1]   = np.dot(np.dot(Z_t.T, inverse(F_t[j])), Z_t) + np.dot(np.dot(L_t[j].T, N_t[j]), L_t[j])
              V_t[j]       = P[j] - np.dot(np.dot(P[j], N_t[j - 1]), P[j]) 
              q005[j] = np.array([[norm.ppf(0.05, loc = float(alpha_hat[j]), scale = ms.sqrt(float(V_t[j])))]])
              q095[j] = np.array([[norm.ppf(0.95, loc = float(alpha_hat[j]), scale = ms.sqrt(float(V_t[j])))]])
             
              # For fig 2.3
              eps_hat[j]   = y[j] - alpha_hat[j]
              D[j]         = inverse(F_t[j]) + np.dot(np.dot(K_t[j], K_t[j]), N_t[j]) # equation 2.47
              var_eps_yn[j]= H_t - np.dot(np.dot(H_t, H_t), D[j])
              eta_hat[j]   = np.dot(Q_t, r_t[j])
              var_eta_yn[j]= Q_t - np.dot(np.dot(Q_t, Q_t), N_t[j]) # equation 2.47
         
      return r_t, alpha_hat, N_t, V_t, q005, q095, eps_hat, var_eps_yn, eta_hat, var_eta_yn, D
  
    
  
    
 
# %% Plotting functions for 2.1 - 2.8

def Plot21(x, y, a, p, v, F,q005, q095, ftsize, lw): 
    """
    Plots the 4 plots of figure 2.1 of Time Series Analysis by State Space Methods bu Durbin J., Koopman S.J.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)
    
    # array of 1x1 arrays to a list of values for t = 2, ... 100
    a = [float(el) for el in a[1:-1]]
    p = [float(el) for el in p[1:-1]]
    v = [float(el) for el in v[1:]]
    F = [float(el) for el in F[1:]]
    q005 = [float(el) for el in q005[1:]]
    q095 = [float(el) for el in q095[1:]]
    
    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.scatter(x, y, color = "black", s = 10)
    ax1.plot(x[1:], a, color = "darkslateblue", lw = lw)
    ax1.plot(x[1:], q005, color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.plot(x[1:], q095, color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)
    
    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.plot(x[1:], p, color = "darkslateblue", lw=lw)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    ax3.plot(x[1:], v, color = "darkslateblue", lw=lw)
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax3.hlines(0,1850,2021, color = 'black', lw = 1)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)
            
    # SUBPLOT 4 below right ----------------------------------------------
    ax4.plot(x[1:], F, color = "darkslateblue", lw=lw)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)



def Plot22(x, y, alpha_hat, V_t, r_t, N_t, q005, q095, ftsize, lw): 
    """
    Plots the 4 plots of figure 2.2 of Time Series Analysis by State Space Methods bu Durbin J., Koopman S.J.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)
    
    alpha_hat = [float(el) for el in alpha_hat]
    V_t       = [float(el) for el in V_t]
    r_t       = [float(el) for el in r_t]
    N_t       = [float(el) for el in N_t]
    q005      = [float(el) for el in q005]
    q095      = [float(el) for el in q095]
    
    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.scatter(x, y, color = "black", s = 10)
    ax1.plot(x, alpha_hat, color = "darkslateblue", lw = lw)
    ax1.plot(x, q005, color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.plot(x, q095, color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)

    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.plot(x, V_t, color = "darkslateblue", lw=lw)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    ax3.plot(x, r_t, color = "darkslateblue", lw=lw)
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax3.hlines(0,1850,2021, color = 'black', lw = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)

    # SUBPLOT 4 below right ----------------------------------------------
    ax4.plot(x, N_t, color = "darkslateblue", lw=lw)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)   
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)




def Plot23(x, y, eps_hat, var_eps_yn, eta_hat, var_eta_yn, ftsize, lw): 
    """
    Plots the 4 plots of figure 2.2 of Time Series Analysis by State Space Methods bu Durbin J., Koopman S.J.
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)

    eps_hat    = [float(el) for el in eps_hat]
    var_eps_yn = [ms.sqrt(float(el)) for el in var_eps_yn] # we plot standard deviations instead of variance
    eta_hat    = [float(el) for el in eta_hat]
    var_eta_yn = [ms.sqrt(float(el)) for el in var_eta_yn] # we plot standard deviations instead of variance 

    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.plot(x, eps_hat, color = "darkslateblue", lw = lw)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax1.hlines(0,1850, 2021, color = 'black', lw = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)

    # # SUBPLOT 2 upper right ----------------------------------------------
    ax2.plot(x, var_eps_yn, color = "darkslateblue", lw=lw)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    ax3.plot(x, eta_hat, color = "darkslateblue", lw=lw)
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax3.hlines(0,1850, 2021, color = 'black', lw = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)

    # SUBPLOT 4 below right ----------------------------------------------
    ax4.plot(x, var_eta_yn, color = "darkslateblue", lw=lw)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)



def Plot25(x, y, a, p, alpha_hat, V_t_mis, ftsize, lw): 
    """
    Plots the 4 plots of figure 2.1 of Time Series Analysis by State Space Methods bu Durbin J., Koopman S.J.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)
    
    # array of 1x1 arrays to a list of values for t = 2, ... 100
    a         = [float(el) for el in a[1:]]
    p         = [float(el) for el in p[1:]]
    alpha_hat = [float(el) for el in alpha_hat]
    V_t_mis   = [float(el) for el in V_t_mis]
    
    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.plot(x, y, color = "black")
    ax1.plot(x, a, color = "darkslateblue", lw = lw)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)
    
    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.plot(x, p, color = "darkslateblue", lw=lw)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    ax3.plot(x, y, color = "black")
    ax3.plot(x, alpha_hat, color = "darkslateblue")
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)
            
    # SUBPLOT 4 below right ----------------------------------------------
    ax4.plot(x, V_t_mis, color = "darkslateblue", lw=lw)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)



def Plot26(x, y, a, q1, q2, P, F_t, ftsize, lw): 
    """
    Plots the 4 plots of figure 2.6 of Time Series Analysis by State Space Methods bu Durbin J., Koopman S.J.
    """
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)
    

    a = [float(el) for el in a[1:]]
    P = [float(el) for el in P[1:]]
    F_t = [float(el) for el in F_t]
    q1_f  = [float(el) for el in np.delete(q1, np.where(q1 == 0))]
    q2_f  = [float(el) for el in np.delete(q2, np.where(q2 == 0))]
    
    
    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.scatter(x, y, color = "black", s = 10)
    ax1.plot(x, a, color = "darkslateblue", lw = 2)
    ax1.plot(np.delete(x, np.where(q1 == 0)), q1_f, color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.plot(np.delete(x, np.where(q1 == 0)), q2_f, color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)

    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.plot(x, P, color = "darkslateblue", lw=lw)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    ax3.plot(x, a, color = "darkslateblue", lw = lw)
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)

    # SUBPLOT 4 below right ----------------------------------------------
    ax4.plot(x[1:], F_t[1:], color = "darkslateblue", lw=lw)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)


def Plot27(x, y, e_t, ftsize, lw): 
    """
    Plots the 4 plots of figure 2.7 of Time Series Analysis by State Space Methods bu Durbin J., Koopman S.J.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)
    e_t = [float(el) for el in e_t]
    
    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.plot(x, e_t, color = "darkslateblue", lw = 2)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax1.hlines(0,1850, 2021, color = 'black', lw = 1)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)

    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.hist(e_t, color = "darkslateblue", lw=lw,facecolor='none', edgecolor="darkslateblue", density = True)
    sns.distplot(e_t, hist=False, ax=ax2)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    sm.qqplot(np.array(e_t), line = 'r', ax =ax3, linestyle='solid', marker  = ',', color ="darkslateblue")
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax3.hlines(0,-2.4, 2.4, color = 'black', lw = 1)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)

    # SUBPLOT 4 below right ----------------------------------------------
    sm.graphics.tsa.plot_acf(e_t, zero = False, lags=10, ax  =ax4, marker  = ',', alpha = None)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)

def Plot28(x, y, u_t_star, r_t_star, ftsize, lw): 
    """
    Plots the 4 plots of figure 2.8 of Time Series Analysis by State Space Methods bu Durbin J., Koopman S.J.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)
    u_t_star = [float(el) for el in u_t_star]
    r_t_star = [float(el) for el in r_t_star]
    
    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.plot(x, u_t_star, color = "darkslateblue", lw = 2)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax1.hlines(0,1850, 2021, color = 'black', lw = 1)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)

    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.hist(u_t_star, color = "darkslateblue", lw=lw,facecolor='none', edgecolor="darkslateblue", density = True)
    sns.distplot(u_t_star, hist=False, ax=ax2)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left ------------------------------------------------
    ax3.plot(x, r_t_star, color = "darkslateblue", lw = 2)
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    # ax3.set_xlim([1864, 1972])
    # ax3.set_ylim([-3.6, 2.4])
    ax3.hlines(0,1850, 2021, color = 'black', lw = 1)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)

    # SUBPLOT 4 below right ----------------------------------------------
    ax4.hist(r_t_star, color = "darkslateblue", lw=lw,facecolor='none', edgecolor="darkslateblue", density = True)
    sns.distplot(r_t_star, hist=False, ax=ax4)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)


def main():
    
    # Get Nile data and convert from to DataFrame to Array
    data  = pd.read_csv ('annual-sea-surface-temp-anomaly.csv')
    data = data.loc[data['Entity'] == 'World']
    x     = np.array(data["Year"])
    y     = np.array(data["annual_sea_surface_temperature_anomaly"])
    y_mis = removedata(removedata(y, 20, 40),60 ,80)
    x_for, y_for = empty_forecast(x, y, n_for = 30)
    
    # First estimate using MLE and then Apply Kalman Filter and Smoother
    sigma2_eps, sigma2_eta = get_MLE()
    print('MLE estimate is (sigma2_eps, sigma2_eta)  = ',(sigma2_eps,sigma2_eta))
    a, P, v_t, F_t, K_t, L_t, q005, q095, n                                                             = Kalman_Filter(y, a1 = 0, p1 = 10 ** 7, sigma2_eps = sigma2_eps, sigma2_eta = sigma2_eta)
    r_t, alpha_hat, N_t, V_t, q005_smooth, q095_smooth, eps_hat, var_eps_yn, eta_hat, var_eta_yn, D_t   = Kalman_Smoother(n, v_t, F_t, L_t, a, P, y, K_t, sigma2_eps = sigma2_eps, sigma2_eta = sigma2_eta)
    
    # Apply Kalman Filter and Smoother: Missing observation case 
    a_mis, P_mis, v_t_mis, F_t_mis, K_t_mis, L_t_mis, q005_mis, q095_mis, n                              = Kalman_Filter(y_mis, a1 = 0, p1 = 10 ** 7, sigma2_eps = sigma2_eps, sigma2_eta = sigma2_eta)
    r_t_mis, alpha_hat_mis, N_t_mis, V_t_mis, q005_mis, q095_mis, eps_hat_mis, var_eps_yn_mis, eta_hat_mis, var_eta_yn_mis, D_t_mis = Kalman_Smoother(n, v_t_mis, F_t_mis, L_t_mis, a_mis, P_mis, y_mis, K_t_mis, sigma2_eps = sigma2_eps, sigma2_eta = sigma2_eta)
    
    # Apply Kalman Filter and Smoother: Forecast case 
    a_f, P_f, v_t_f, F_t_f, K_t_f, L_t_f, q1_f, q2_f, n_f = Kalman_Filter_Forecast(y_for, a1 = 0, p1 = 10 ** 7, sigma2_eps = sigma2_eps, sigma2_eta = sigma2_eta)
    
    # get standardized residuals for figure 2.7 and 2.8
    e_t = v_t / np.sqrt(F_t)
    u_t      = eps_hat / sigma2_eps
    u_t_star = (D_t ** ((-1) / 2)) * u_t
    r_t_star = (N_t ** ((-1) / 2)) * r_t
    
    
    # Plot Figures 
    Plot21(x, y, a, P, v_t, F_t, q005, q095, ftsize = 12, lw = 1.5)
    Plot22(x, y, alpha_hat, V_t, r_t, N_t, q005_smooth, q095_smooth, ftsize = 12, lw = 1.5)
    Plot23(x, y, eps_hat, var_eps_yn, eta_hat, var_eta_yn, ftsize = 12, lw = 1.5)
    Plot25(x, y_mis, a_mis, P_mis, alpha_hat_mis, V_t_mis, ftsize = 12, lw = 1.5)
    Plot26(x_for, y_for, a_f, q1_f, q2_f, P_f, F_t_f, ftsize = 12, lw = 1.5)
    Plot27(x, y, e_t, ftsize = 12, lw  = 1.5)
    Plot28(x, y, u_t_star, r_t_star, ftsize = 12, lw = 1.5)


if __name__ == '__main__':
     main()


































