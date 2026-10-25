import sys, time, json; sys.path.insert(0,'tools')
import numpy as np, torch
import gpu_mprune as G
src=json.load(open('logs/bonus_export_10_3.json'))
N,n,K=src['N'],src['n'],len(src['qubits'])
edges=sorted({tuple(sorted((o[1],o[2]))) for mo in src['moments'] for o in mo if o[0]=='cz'})
M=len(src['moments']); masks=np.zeros((M,len(edges)),int); P=np.zeros((M,K,3))
for m,mo in enumerate(src['moments']):
    for o in mo:
        if o[0]=='cz': masks[m,edges.index(tuple(sorted((o[1],o[2]))))]=1
        else: P[m,o[1]]=o[2]
c=G.Circ(K,edges,masks,P); tgt=G.dicke_vec(N,n)
t=time.time(); d,f=G.build_torch(K,edges,[masks]*64,torch.float32); torch.cuda.synchronize(); print('build',time.time()-t, flush=True)
t=time.time(); G.optimise([c]*64,K,edges,tgt,20); torch.cuda.synchronize(); print('20 steps B=64',time.time()-t, flush=True)
t=time.time(); cc=G.delete_moment(c,5,False); print('delete_moment',time.time()-t, flush=True)
