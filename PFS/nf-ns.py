import os, sys 
import numpy as np
ns = np.array(range(2,18))
nf = np.array(range(2,20))
for ins in ns:
    for inf in nf:
        if os.path.exists(f"E-f{inf}-n{ins}-η0.2.txt"):
            print (f"E-f{inf}-n{ins}-η0.2.txt exists")
        else:
            os.system(f"sbatch parallel.py {inf} {ins}")
