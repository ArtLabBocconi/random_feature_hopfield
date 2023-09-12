import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("magnetisations_aD0.01.txt")
theory = np.loadtxt("../saddle_point/factor/results/span_aD0.01.txt",usecols=(0,10))

plt.figure(figsize=(4,3))

plt.plot(data[:,0], data[:,3], label="feature mag", marker="v")
plt.plot(data[:,0], data[:,2], label="pattern mag", marker="o")
plt.plot(theory[:,0], theory[:,1], label="theory", color="black", linestyle="dashed")

plt.legend(title=r"$\alpha_D=0.01$")
plt.xlabel(r"$\alpha=P/N$")
plt.ylabel("magnetisation")

plt.tight_layout()

plt.savefig("magnetisations_aD0.01.pdf")