#!/software/anaconda3/2020.07/bin/python
#SBATCH -p action 
#SBATCH -o output.log
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1:00:00
#SBATCH -n 24
#SBATCH -N 1
import numpy as np
from multiprocessing import Pool
import time , sys, os
sys.path.append(os.popen("pwd").read().replace("\n",""))
from polariton import Ĥ, param
#-------------------------------------
R0 = 0.0
η  = 0.15 # np.arange(0.0,0.3,0.01)
#-------------------------------------
nf = range(4,21) #param.nf
ns = 2 #param.ns
#-------------------------------------
t0 = time.time()
#----------------------------  SBATCH  ---------------------------------------------------
sbatch = [i for i in open('parallel.py',"r").readlines() if i[:10].find("#SBATCH") != -1 ]
cpu   = int(sbatch[-2].split()[-1].replace("\n","")) #int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
nodes = int(sbatch[-1].split()[-1].replace("\n",""))
print (os.environ['SLURM_JOB_NODELIST'], os.environ['SLURM_JOB_CPUS_PER_NODE'])
print (f"nodes : {nodes} | cpu : {cpu}")
procs = cpu * nodes
#-------------------------------------
E = np.loadtxt("../Hm/E.txt")
N = len(E[0,1:])
R = E[:,0]
E = E[:,1:]
µ = np.loadtxt("../Hm/µ.txt")[:,1:]
Ridx = np.argmin(np.abs(R - R0))
#-------------------------------------
ns = min(ns,N)
with Pool(len(R)) as p:
    #------ Arguments for each CPU--------
    args = []
    for j in range(len(nf)):
        par = param()
        par.ns = ns
        par.nf = nf[j]
        par.η = η
        par.χ = par.ωc * η 
        #----------------------
        H = np.zeros((N, N))
        H[np.diag_indices(N)] = E[Ridx,:]
        par.H = H[:ns,:ns]
        par.µ = µ[Ridx,:].reshape((N,N))[:ns,:ns]
        Hij = Ĥ(par)
        args.append(Hij)

    #-------- parallelization --------------- 
    result  = p.map(np.linalg.eigh, args)
    #----------------------------------------
t2 = time.time() - t0 
print (f"Time taken: {t2} s")
print (f"Time for each point: {t2/len(R)} s")
#------- Gather -----------------------------
E, _ = result[0]
nstate = 8
E = np.zeros((nstate+1, len(nf)))

for ηi in range(len(nf)):
    E[1:,ηi] = result[ηi][0][:nstate]
#---- Shift ZPE ----------
#E0 = np.min(E[1,:]) # ZPE
E[1:,:] -= E[1,:]
#-------------------------
E[0,:] = np.array(nf)
#--------------------------------------------
np.savetxt(f"E-n{ns}-η{η}-R{R0}.txt", E.T)