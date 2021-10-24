import numpy as np
def olap(N, w, R0, conv = 30, M =1):
    R0 =float(R0)
    # nbasis
    nbasis = 5* N + conv
    Hij = np.zeros((nbasis,nbasis))  
    for m in range(nbasis):
        for n in range(nbasis):
            Hij[m,n] = (n * w) * float(m==n) 
            const1 = M * (w**2) * R0 * ((1.0/(2.0 * M * w))**0.5) 
            Hij[m,n] += const1 * ( ((float(m))**0.5) * float(n==(m-1)) +  ((float(m+1))**0.5) * float(n==(m+1))  )
            const2 = 0.5 * M * (w**2) * R0 ** 2.0
            Hij[m,n] += const2 * float(m==n)
    E , V = np.linalg.eigh(Hij) 
    err =  (E[N]/w - N ) * 100.0 /float(N) 
    if abs(err) > 1E-8 : 
        print ("Overlap may not be converged!" )
    return V[:N,:N] 

if __name__ == "__main__":
    for i in range(1,10):
        V = olap(7, 3.0/27.2114, i) 
        np.savetxt("olap%s.txt"%(i),V)

