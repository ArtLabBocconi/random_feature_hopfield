# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data_mu = np.loadtxt("../factor/alphac_factor.txt", usecols=(0,1),encoding='utf-8')
data_m2 = np.loadtxt("../pattern/alphac_pattern.txt", usecols=(0,1),encoding='utf-8')
data_m2 = data_m2[np.argsort(data_m2[:,1]),:]

# data_mu = np.loadtxt("../GET_calculation_naive/codes_saddle_point/hopfield/alphac.txt", usecols=(0,1),encoding='utf-8')
# data_m2 = np.loadtxt("../GET_calculation_naive/PatternMagnetization/Code/alphac_clean.txt", usecols=(0,1),encoding='utf-8')
# data_m2 = data_m2[np.argsort(data_m2[:,1]),:]

# np.savetxt("alphac_f.txt",data_mu)
# np.savetxt("alphac_xi.txt",data_m2)
# np.savetxt("PatternMagnetization/Code/alphac_prun_sorted.txt", data_m2)
###############################################################

fig, ax1 = plt.subplots(figsize=(4,3))

plt.plot(data_mu[:,1], data_mu[:,0], label=r"$f-recovery threshold$")
# plt.plot(data_m[:,1] , data_m[:,0] , label="xi spinodal")
plt.plot(data_m2[:,1], data_m2[:,0], label=r"$\xi-recovery threshold$")

plt.vlines(0.138, 0, 2, linestyles="dashed", linewidths=0.3)
plt.hlines(0.138, 0, 1, linestyles="dashed", linewidths=0.3)

plt.xlim(0,1)
plt.ylim(0,1)

# plt.legend()
plt.xlabel(r"$\alpha_D=D/N$")
plt.ylabel(r"$\alpha=P/N$")
###############################################################
plt.text(0.15, 0.9, "Learning")#,fontweight="bold")
plt.arrow(0.12, 0.92, -0.08, 0, head_width=0.01)

plt.text(0.8, 0.2, "Storage")#,fontweight="bold")
plt.arrow(0.88, 0.18, 0, -0.11, head_width=0.01)

###############################################################
# data_sim = np.loadtxt("alphac_num_small_aD.txt")
# plt.plot(data_sim[:,0], data_sim[:,1], '-')

###############################################################
ax2 = fig.add_axes([0.49, 0.49, 0.4, 0.4])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.plot(data_mu[:,1], data_mu[:,0], label=r"$f$-recovery"+"\nthreshold")
# plt.plot(data_m[:,1] , data_m[:,0] , label="xi spinodal")
plt.plot(data_m2[:,1], data_m2[:,0], label=r"$\xi$-recovery"+"\nthreshold")

# plt.plot(data_sim[:,0], data_sim[:,1], 'x-')

###############################################################
max_i = 6

def func1(x, a,):
    return a * x

x_to_fit = data_mu[:max_i,1]
y_to_fit = data_mu[:max_i,0]
x_to_plot = np.linspace(0, 0.01, 200)

popt, pcov = curve_fit(func1, x_to_fit, y_to_fit)
#plt.plot(x_to_plot, func1(x_to_plot, *popt),"--", linewidth=1, color="C0")#, label='{:1.2f}'.format(*popt))


###############################################################
max_i = 6

def func2(x, a):
    return a * x**(1/2) 

def func3(x):
    return 0.08 * x**(1/2) 

x_to_fit = data_m2[:max_i,1]
y_to_fit = data_m2[:max_i,0]
x_to_plot = np.linspace(0, 0.05, 1000)

popt, pcov = curve_fit(func2, x_to_fit, y_to_fit)
#plt.plot(x_to_plot, func2(x_to_plot, *popt),"--", linewidth=1, color="C1")

# plt.xlim(0,0.04)
plt.xlim(0,0.02)
plt.ylim(0,0.05)

# plt.legend(fontsize=8)
ax2.legend(fontsize=9,loc='lower right',bbox_to_anchor=(1.12, 0))
# ax2.legend()

###############################################################
plt.tight_layout()
plt.savefig("fig_1A.pdf")
# %%
