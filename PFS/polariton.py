import math,mpmath
import time
import numpy as np
import scipy as sp
import sys
from numpy import linalg as LA
from numpy import kron as ꕕ
from numpy import array as A
from olap import olap as D
from scipy.special import hermite
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
    #H = A([[0,0],[0,1]])
    #µ = A([[0,1],[1,0]])
    ns = 2
    nf = 4
    η  = 0.15
    ωc = 0.4/27.2114
    χ = ωc * η

#----------------------------------------
def HO(x,w,n):
    m = 1
    cons1 = 1.0/((2.0**n) * math.factorial(n))**0.5
    cons2 = ( m * w / (math.pi))**0.25 
    exp  = np.exp(- (m*w*(x**2)/2.0)) 
    cons3 = ((m*w)**0.5)
    #cons4 = cons3 * (2**0.5)
    hermit = hermite(n)(cons3 * x) #mpmath.hermite(n,cons3 * x)
    H =  hermit
    val = cons1 * cons2 * exp * H 
    return val


def qc0(µii, param):
    return -param.χ * µii * (2.0/param.ωc**3.0)**0.5
#----------------------------------------
# Data of the diabatic states

def Ĥ(param):
    H = param.H * 1
    µ = param.µ * 1
    #----- MH Basis --------------------
    µii, Uµ = np.linalg.eigh(µ)
    Hµ = Uµ.T @ H @ Uµ
    #-----------------------------------

    ns = param.ns
    nf = param.nf
    ωc = param.ωc 
    χ = param.χ
    dr = 0.005
    rc = np.arange(-50,50,dr)
    
    HI = np.zeros((nf*ns,nf*ns))
    for i in range(nf*ns):
        a = int(i/nf)
        m = i%nf
        for j in range(i,nf*ns):
            b = int(j/nf)
            n = j%nf
            smn = np.sum(HO(rc-qc0(µii[a], param),ωc,m) * HO(rc-qc0(µii[b], param),ωc,n)) * dr
            HI[i,j]  = Hµ[a,b]  * (1-(a==b)) * smn
            HI[i,j] += (m + 0.50000) * ωc * (m==n) * (a==b)
            HI[i,j] += Hµ[a,b] * (a==b) * (m==n)
            HI[j,i] = HI[i,j]
    return HI
#--------------------------------------------------------

#----------------------------------------

if __name__ == "__main__":

 E = np.loadtxt("../Hm/E.txt")
 µ = np.loadtxt("../Hm/µ.txt")[:,1:]
 R = E[:,0]
 N = len(E[0,1:])
 H   = np.zeros((N, N))
 H[np.diag_indices(N)] = E[len(R)//2,1:]


 param.H = H[:param.ns,:param.ns]
 param.µ = µ[len(R)//2,:].reshape((N,N))[:param.ns,:param.ns]
 Hij = Ĥ(param)
 print (Hij)
 np.savetxt("Hij.txt",Hij)





