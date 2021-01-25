"""
This script requires matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
from rlberry.agents.kernel_based.kernels import kernel_func

kernel_types = [
                "uniform",
                "triangular",
                "gaussian",
                "epanechnikov",
                "quartic",
                "triweight",
                "tricube",
                "cosine",
                "exp-4"
                ]

z = np.linspace(-2, 2, 100)

fig, axes = plt.subplots(1, len(kernel_types))
for ii, k_type in enumerate(kernel_types):
    kernel_vals = kernel_func(z, k_type)
    axes[ii].plot(z, kernel_vals)
    axes[ii].set_title(k_type)
plt.show()
