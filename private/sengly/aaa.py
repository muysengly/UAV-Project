# %%

import numpy as np

from sklearn.cluster import KMeans

# %%


gu_xyz = np.random.random(size=(10, 3))
gu_xyz

# %%

aa = KMeans(
    n_clusters=7,
    n_init="auto"
).fit(gu_xyz)

