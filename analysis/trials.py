import numpy as np, math
import score_analysis as sa
def trials(N,n,K,a0,a1,q,bias,T):
    out=[]
    for _ in range(T):
        zs,xs,f,s = sa.run(N,n,K,a0,a1,q,bias)
        out.append((zs,xs,f,s))
    a=np.array(out)
    return a
print("N n M | scenario | median overall | mean overall | p90 overall | median Z | median X | f")
for N,n in [(2,1),(3,1),(4,1),(4,2),(6,1),(6,3),(8,4),(10,5)]:
    M=math.comb(N,n); K=min(N,4)
    for name,(a0,a1,q,b) in {"ideal":(0,0,0,0),"readout":(0.005,0.01,0.0075,0),
                             "ro+leak2%":(0.015,0.03,0.03,0),"ro+bias2%":(0.005,0.01,0.0075,0.02)}.items():
        T=150 if M<30 else 40
        a=trials(N,n,K,a0,a1,q,b,T)
        print(f"{N} {n} {M:3d} | {name:9s} | {np.median(a[:,3]):6.2f} | {np.mean(a[:,3]):7.2f} | {np.percentile(a[:,3],90):6.2f} | {np.median(a[:,0]):.3f} | {np.median(a[:,1]):.4f} | {np.median(a[:,2]):.3f}")
