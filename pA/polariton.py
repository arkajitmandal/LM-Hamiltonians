import math
import time
import numpy as np
import scipy as sp
import sys
from numpy import linalg as LA
from numpy import kron as ꕕ
from numpy import array as A
#global xi
#xi = float(sys.argv[1])

def Eig(H):
    E,V = LA.eigh(H) # E corresponds to the eigenvalues and V corresponds to the eigenvectors
    return E,V
#-------------------------------------
#-------------------------------------
def ĉ(nf):
    a = np.zeros((nf,nf))
    for m in range(1,nf):
        a[m,m-1] = np.sqrt(m)
    return a.T

class param:
    H = A([[0,0],[0,1]])
    µ = A([[0,1],[1,0]])
    ns = 2
    nf = 8
    η  = 0.15
    ωc = 0.4/27.2114
    χ = ωc * η

#----------------------------------------
# Data of the diabatic states

def Ĥ(param):
    Hm = param.H.astype("complex")
    µ = param.µ
    ns = param.ns
    nf = param.nf
    #--------Delete Permanent Dipole -------------
    µ[np.diag_indices(param.ns)] = 0
    p = np.zeros(µ.shape,dtype='complex')
    for i in range(ns):
        for j in range(i+1,ns):
            p[i,j] = 1j * µ[i,j] * (Hm[i,i] - Hm[j,j])
            p[j,i] = - p[i,j]

    #---------------------------------------------
    ωc = param.ωc 
    χ = param.χ
    #------------------------
    Iₚ = np.identity(nf).astype("complex")
    Iₘ = np.identity(ns).astype("complex")
    #------ Photonic Part -----------------------
    Hₚ = np.identity(nf).astype("complex")
    Hₚ[np.diag_indices(nf)] = np.arange(nf) * ωc
    â = ĉ(nf) 
    #------ Vector Potential --------------------
    A0 = χ/ωc
    Â = A0 * (â.T + â)
    #--------------------------------------------
    #       matter ⊗ photon 
    #--------------------------------------------
    Hij   = ꕕ(Hm, Iₚ)            # Matter
    Hij  += ꕕ(Iₘ, Hₚ)            # Photon
    Hij  -= ꕕ(p, Â)              # Interaction
    Hij  += ꕕ(Iₘ, Â @ Â) * 0.5   # Diamagnetic term
    return Hij 
#--------------------------------------------------------

#----------------------------------------

if __name__ == "__main__":
 print (Ĥ(param))






