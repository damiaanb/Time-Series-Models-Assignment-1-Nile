import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import norm
import math as ms 

    
def Kalman(y, a1, p1, sigma2_eps, sigma2_eta):

    a = [a1]
    p = [p1]
    v = [] 
    F = []
    q005 = []
    q095 = []
    
    for i in range(len(y)):
        v   += [y[i] - a[-1]]
        F   += [p[-1] + sigma2_eps]
        k_t  = p[-1] / (p[-1] + sigma2_eps)
        a_tt = a[-1] + k_t * v[-1]
        p_tt = k_t * sigma2_eps
        q005 += [norm.ppf(0.05, loc = a_tt, scale = ms.sqrt(p_tt))]
        q095 += [norm.ppf(0.95, loc = a_tt, scale = ms.sqrt(p_tt))]
        a   += [a_tt]
        p   += [p_tt + sigma2_eta]
    
    a.pop(0)
    p.pop(0)
    v.pop(0)
    F.pop(0)
    
    return a, p, v, F, q005, q095



def Plot4Subplots(x, y, a, p, v, F, q005, q095, ftsize, lw):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
    fig.set_size_inches(12, 8)
    
    
    # # Set the ticks and ticklabels for all axes
    # plt.setp(((ax1, ax2), (ax3, ax4)), xticks=[1880,1900], xticklabels=['ajax', 'b'])

    # SUBPLOT 1 upper left ------------------------------------------------
    ax1.scatter(x, y, color = "black", s = 10)
    ax1.plot(x.drop(0), a[:-1], color = "darkslateblue", lw = lw)
    ax1.plot(x.drop(0), q005[:-1], color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.plot(x.drop(0), q095[:-1], color = "mediumslateblue", lw = 1, alpha =0.6)
    ax1.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    # ax1.set_xticks(ticks = [1870 + i * 10 for i in range(11)], labels = ["1870","1880","1890","1900","1910","1920","1930","1940","1950","1960","1970"]) 
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax1.spines[axis].set_linewidth(lw)
        else:
            ax1.spines[axis].set_visible(False)
    
    # SUBPLOT 2 upper right ----------------------------------------------
    ax2.plot(x, p, color = "darkslateblue", lw=lw)
    ax2.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)
    ax2.set_xlim([1860, 1970])
    ax2.set_ylim([5000, 17500])
    
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax2.spines[axis].set_linewidth(lw)
        else:
            ax2.spines[axis].set_visible(False)
    
    # SUBPLOT 3 below left -----------------------------------------------
    ax3.plot(x.drop(0), v, color = "darkslateblue", lw=lw)
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
    ax4.plot(x.drop(0), F, color = "darkslateblue", lw=lw)
    ax4.tick_params(axis='both', which='major', labelsize=ftsize, width = lw)

    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax4.spines[axis].set_linewidth(lw)
        else:
            ax4.spines[axis].set_visible(False)



def main():
    data = pd.read_excel("Nile.xlsx", names =["year", "volume"])
    x = data["year"]
    y = data["volume"]
    
    a, p, v, F, q005, q095 = Kalman(y, a1 = 0, p1 = 10 ** 7, sigma2_eps = 15099, sigma2_eta = 1469.1)
    Plot4Subplots(x, y, a, p, v, F, q005, q095, ftsize = 12, lw = 1.5)




if __name__ == '__main__':
    
    main()






