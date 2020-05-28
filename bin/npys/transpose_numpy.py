import numpy as np
import sys
a = np.load(sys.argv[1])
print(a.shape)
np.save(sys.argv[1],a.transpose([2,0,1]))
