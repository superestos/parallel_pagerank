import sys
import numpy as np
import pandas as pd

origin = sys.argv[1]
processed = sys.argv[2]

df = pd.read_csv(origin, sep='\t', header=None, comment='#')
coo = df.to_numpy()
nodes = np.unique(coo)

reorder = {}
for i in range(nodes.shape[0]):
    reorder[nodes[i]] = i
    
for i in range(coo.shape[0]):
    coo[i][0] = reorder[coo[i][0]]
    coo[i][1] = reorder[coo[i][1]]
    
coo = coo[coo[:,1].argsort()]

coo.T.astype(np.int32).tofile(processed)
