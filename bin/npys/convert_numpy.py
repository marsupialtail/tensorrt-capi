import numpy as np
import sys
a = np.load(sys.argv[1])
np.save(sys.argv[1],a.astype(np.float16))
