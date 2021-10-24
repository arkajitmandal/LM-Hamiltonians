import numpy as np 
import os
Rl = np.linspace(-4,4,40)
os.system("rm -rf R-*")
for iRl in range(len(Rl)-1):
    Rmn = Rl[iRl]
    point = 5
    Rmx = np.linspace(Rl[iRl],Rl[iRl+1],point+1)[-2]
    os.mkdir(f"R-{iRl}")
    os.chdir(f"R-{iRl}")
    os.system("cp ../SM.py ./")
    os.system("cp ../parallel.py ./")
    os.system(f"sbatch parallel.py {Rmn} {Rmx} {point}")
    os.chdir(f"../")
