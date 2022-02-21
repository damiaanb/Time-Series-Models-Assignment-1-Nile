import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import norm
import math as ms 
import numpy as np
    

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
        
        
        v_t[i]  = y[i] - np.dot(Z_t, a[i])
        F_t[i]  = np.dot(np.dot(Z_t, P[i]), Z_t.T) + H_t
        K_t[i]  = np.dot(np.dot(np.dot(T_t, P[i]), Z_t.T), inverse(F_t[i]))
        L_t[i]  = T_t - np.dot(K_t[i], Z_t)
        q005[i] = np.array([[norm.ppf(0.05, loc = float(a[i]), scale = ms.sqrt(float(P[i])))]])
        q095[i] = np.array([[norm.ppf(0.95, loc = float(a[i]), scale = ms.sqrt(float(P[i])))]])
        a[i + 1]   = np.dot(T_t, a[i]) + np.dot(K_t[i], v_t[i])
        P[i + 1]   = np.dot(np.dot(T_t, P[i]), T_t.T) + np.dot(np.dot(R_t, Q_t), R_t.T) - np.dot(np.dot(K_t[i], F_t[i]), K_t[i].T)

    return a, P, v_t, F_t, K_t, L_t, q005, q095, n



def  Kalman_Smoother(n, v_t, F_t, L_t, a, P):
    
     Z_t = np.array([[1]]) 
    
     r_t        = np.zeros((n,1,1))
     r_t[n - 1] = np.array([[0]])
     alpha_hat  = np.zeros((n,1,1))
     
     N_t        = np.zeros((n,1,1))
     N_t[n - 1] = np.array([[0]])
     V_t        = np.zeros((n,1,1))
     V_t[n - 1] = np.array([[0]])
     
     
     for j in range(n-1,-1,-1):
         
         r_t[j - 1]   = np.dot(v_t[j], inverse(F_t[j])) + np.dot(L_t[j], r_t[j])
         alpha_hat[j] = a[j] + np.dot(P[j], r_t[j - 1])
          
         N_t[j - 1]   = np.dot(np.dot(Z_t.T, inverse(F_t[j])), Z_t) + np.dot(np.dot(L_t[j].T, N_t[j]), L_t[j])
         V_t[j]       = P[j] - np.dot(np.dot(P[j], N_t[j - 1]), P[j])
         
     return r_t, alpha_hat, N_t, V_t
 


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
    ax1.set_ylim([450, 1400])
    # ax1.set_xticks(ticks = [1870 + i * 10 for i in range(11)], labels = ["1870","1880","1890","1900","1910","1920","1930","1940","1950","1960","1970"]) 
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)
    
    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.plot(x[1:], p, color = "darkslateblue", lw=lw)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax2.set_xlim([1860, 1970])
    ax2.set_ylim([5000, 17500])
    
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    ax3.plot(x[1:], v, color = "darkslateblue", lw=lw)
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    #ax3.xticks([1880,1900],["akax", "fey"], rotation='vertical')
    ax3.set_xlim([1862, 1972])
    ax3.set_ylim([-450, 400])
    ax3.hlines(0,1860, 1970, color = 'black', lw = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)
            
    # SUBPLOT 4 below right ----------------------------------------------
    ax4.plot(x[1:], F, color = "darkslateblue", lw=lw)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax4.set_xlim([1860, 1970])
    ax4.set_ylim([20000, 32500])
    
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)



def Plot22(x, y, alpha_hat, V_t, r_t, N_t, ftsize, lw): 
    """
    Plots the 4 plots of figure 2.2 of Time Series Analysis by State Space Methods bu Durbin J., Koopman S.J.
    """
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)
    

    alpha_hat = [float(el) for el in alpha_hat[1:-1]]
    V_t       = [float(el) for el in V_t]
    r_t       = [float(el) for el in r_t]
    N_t       = [float(el) for el in N_t]
    
    
    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.scatter(x, y, color = "black", s = 10)
    ax1.plot(x[2:], alpha_hat, color = "darkslateblue", lw = lw)
    # ax1.plot(x[1:], q005, color = "mediumslateblue", lw = 1, alpha =0.6)
    # ax1.plot(x[1:], q095, color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax1.set_ylim([450, 1400])
    # ax1.set_xticks(ticks = [1870 + i * 10 for i in range(11)], labels = ["1870","1880","1890","1900","1910","1920","1930","1940","1950","1960","1970"]) 
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)

    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.plot(x, V_t, color = "darkslateblue", lw=lw)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax2.set_xlim([1860, 1970])
    ax2.set_ylim([2300, 4100])
        
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    ax3.plot(x, r_t, color = "darkslateblue", lw=lw)
    ax3.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    #ax3.xticks([1880,1900],["akax", "fey"], rotation='vertical')
    ax3.set_xlim([1862, 1972])
    ax3.set_ylim([-0.036, 0.024])
    ax3.hlines(0,1860, 1970, color = 'black', lw = lw)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax3.spines[axis].set_linewidth(lw)
        else:
            ax3.spines[axis].set_visible(False)

    # SUBPLOT 4 below right ----------------------------------------------
    ax4.plot(x, N_t, color = "darkslateblue", lw=lw)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    # ax4.set_xlim([1860, 1970])
    ax4.set_ylim([5 *(10 ** (-5)), 1.1 * (10 ** (-4))])
    
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)






def main():
    data = pd.read_excel("Nile.xlsx", names =["year", "volume"])
    x = np.array(data["year"])
    y = np.array(data["volume"])
    
    a, P, v_t, F_t, K_t, L_t, q005, q095, n = Kalman_Filter(y, a1 = 0, p1 = 10 ** 7, sigma2_eps = 15099, sigma2_eta = 1469.1)
    r_t, alpha_hat, N_t, V_t                = Kalman_Smoother(n, v_t, F_t, L_t, a, P)
    
    
    Plot21(x, y, a, P, v_t, F_t, q005, q095, ftsize = 12, lw = 1.5)
    Plot22(x, y, alpha_hat, V_t, r_t, N_t, ftsize = 12, lw = 1.5)
    
    return x, y, a, P, v_t, F_t, K_t, L_t, q005, q095, n, r_t, alpha_hat, V_t, N_t



if __name__ == '__main__':
    
    x, y, a, P, v_t, F_t, K_t, L_t, q005, q095, n, r_t, alpha_hat, V_t, N_t = main()






