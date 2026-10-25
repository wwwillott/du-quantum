"""Full calibrated-noise distributions and cheap, exact-formula score draws."""
from dataclasses import replace
import math
import cirq
import cirq_google
import numpy as np
from judges_audit.rescore import score_batch


def model(coherent_scale=1.):
    properties=cirq_google.engine.load_device_noise_properties('willow_pink')
    if coherent_scale!=1:
        gates={key:cirq.PhasedFSimGate(theta=g.theta*coherent_scale,zeta=g.zeta*coherent_scale,
            chi=g.chi*coherent_scale,gamma=g.gamma*coherent_scale,phi=g.phi*coherent_scale)
            for key,g in properties.fsim_errors.items()}
        properties=replace(properties,fsim_errors=gates)
    return cirq_google.NoiseModelFromGoogleNoiseProperties(properties)


def properties_fixed_stochastic(coherent_scale):
    """Change coherent angles while holding thermal/depolarizing noise fixed.

    GoogleNoiseProperties normally subtracts coherent infidelity from the
    supplied total Pauli budget. Freezing the cached residual is essential:
    otherwise removing coherent error replaces it with extra depolarization.
    This version-specific experimental ablation does not change the baseline.
    """
    properties=cirq_google.engine.load_device_noise_properties('willow_pink')
    residual=dict(properties._depolarizing_error)
    gates={key:cirq.PhasedFSimGate(theta=g.theta*coherent_scale,zeta=g.zeta*coherent_scale,
        chi=g.chi*coherent_scale,gamma=g.gamma*coherent_scale,phi=g.phi*coherent_scale)
        for key,g in properties.fsim_errors.items()}
    properties=replace(properties,fsim_errors=gates)
    properties.__dict__['_depolarizing_error']=residual
    return properties


def model_fixed_stochastic(coherent_scale):
    return cirq_google.NoiseModelFromGoogleNoiseProperties(properties_fixed_stochastic(coherent_scale))


def probabilities(circuit,outputs,noise,dtype=np.complex64):
    order=sorted(outputs)+sorted(circuit.all_qubits()-set(outputs))
    distributions={}
    for basis in ('z','x'):
        measured=cirq.Circuit(circuit)
        if basis=='x':
            h=cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))
            measured+=cirq.Circuit(h(q) for q in sorted(outputs))
        measured+=cirq.Circuit(cirq.measure(*sorted(outputs),key='m'))
        expanded=cirq.Circuit(noise.noisy_moments(measured,order))
        expanded=cirq.Circuit(op for op in expanded.all_operations() if not cirq.is_measurement(op))
        rho=cirq.DensityMatrixSimulator(dtype=dtype).simulate(expanded,qubit_order=order).final_density_matrix
        p=np.real(np.diag(rho)).reshape(1<<len(outputs),-1).sum(axis=1).astype(float)
        p=np.maximum(p,0);p/=p.sum()
        distributions[basis]=p
    return distributions


def stats(distributions,n,k,repetitions=64,seed=17):
    result={}
    for max_k in sorted({n,min(4,n)}):
        rng=np.random.default_rng(seed+1000*n+31*k+max_k)
        z=rng.multinomial(10000,distributions['z'],size=repetitions)
        x=rng.multinomial(10000,distributions['x'],size=repetitions)
        scores=score_batch(z,x,n,k,max_k)
        result[str(max_k)]=dict(median=float(np.median(scores)),mean=float(np.mean(scores)),
            q025=float(np.quantile(scores,.025)),q975=float(np.quantile(scores,.975)))
    return result


def penalties(distributions,n,k):
    shell=distributions['z'][[i.bit_count()==k for i in range(1<<n)]]
    f=float(shell.sum());q=shell/f;m=math.comb(n,k)
    return dict(retention=f,finite_shot_Z=(1-float(q@q))/f,
                nonuniform_Z=10000*float(np.sum((q-1/m)**2)))


def expected_denominator(distributions,n,k,max_k=None):
    """Expected official denominator: exact X moments and binomial Z inverse.

    For valid count B, E[Z|B]=(1-sum q^2)*S/B+S*sum(q-1/M)^2.
    Sum the binomial reciprocal numerically, conditioning away B=0, which
    has negligible probability in these high-retention experiments.
    """
    from scipy.stats import binom
    max_k=n if max_k is None else max_k
    shots=10000
    shell=distributions['z'][[i.bit_count()==k for i in range(1<<n)]]
    f=float(shell.sum());q=shell/f;m=math.comb(n,k)
    b=np.arange(1,shots+1)
    inverse=float(np.dot(binom.pmf(b,shots,f),shots/b)/(1-binom.pmf(0,shots,f)))
    z=(1-float(q@q))*inverse+shots*float(np.sum((q-1/m)**2))
    weights=np.array([i.bit_count() for i in range(1<<n)])
    px=distributions['x'];x=0.
    def comb(a,b):return math.comb(a,b) if 0<=b<=a else 0
    for ell in range(1,max_k+1):
        values_by_weight=np.array([sum((-1)**j*comb(w,j)*comb(n-w,ell-j) for j in range(ell+1))/comb(n,ell) for w in range(n+1)])
        values=values_by_weight[weights]
        measured=float(px@values);variance=max(0.,float(px@(values*values))-measured**2)
        ideal=comb(ell,ell//2)*comb(n-ell,k-ell//2)/comb(n,k) if ell%2==0 else 0.
        x+=((ideal-measured)**2+variance/shots)/(ideal**2 if ideal else 1.)
    return float(z+x)
