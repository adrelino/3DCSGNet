import deepdish as dd
from src.Utils.train_utils import voxels_from_expressions

# pre-rendered shape primitives in the form of voxels for better performance
primitives = dd.io.load("data/primitives.h5")
expressions = ["cy(48,48,32,8,12)cu(24,24,40,28)+"]#, "sp(48,48,32,8,12)cu(24,24,40,28)+"]

voxels = voxels_from_expressions(expressions, primitives, max_len=7)
print(voxels.shape)

import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(np.squeeze(voxels), edgecolor='k')
plt.show()