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
    ns = 18
    nf = 18
    η  = 0.15
    ωc = 0.4/27.2114
    χ = ωc * η

#----------------------------------------
# Data of the diabatic states
def Ĥ(param):
    H = param.H 
    µ = param.µ
    ns = param.ns
    nf = param.nf
    ωc = param.ωc 
    χ = param.χ
    #------------------------
    Iₚ = np.identity(nf)
    Iₘ = np.identity(ns)
    #------ Photonic Part -----------------------
    Hₚ = np.identity(nf)
    Hₚ[np.diag_indices(nf)] = np.arange(nf) * ωc
    â   = ĉ(nf) 
    #--------------------------------------------
    #       matter ⊗ photon 
    #--------------------------------------------
    Hij   = ꕕ(H, Iₚ)                    # Matter
    Hij  += ꕕ(Iₘ, Hₚ)                   # Photon
    Hij  += ꕕ(µ, (â.T + â)) * χ         # Interaction
    Hij  += ꕕ(µ @ µ, Iₚ) * (χ**2/ωc)    # Dipole Self-Energy
    return Hij 
#--------------------------------------------------------

#----------------------------------------

if __name__ == "__main__":
 print (Ĥ(param))






