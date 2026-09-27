"""Independent checks of the reconstructed oracle and crossing search.

Run after exp_landscape.py. The saved experiment already checks 3915
crossings. Here we additionally test actual oracle evaluations, KKT
solutions, the endpoint constant, and random branch configurations.
"""
import csv
import json
from pathlib import Path
import numpy as np
import exp_landscape as e

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(831027)
summary = {}

def project(chat):
    x = -chat/e.MU
    return x*min(1.,e.R/np.linalg.norm(x)) if np.linalg.norm(x) else x

def true_value(x, W, p, c, L0):
    return .5*e.MU*(x@x)+c@x+.5*L0*np.sum((W.T@(x-p))**2)

# Actual noisy values on a simplex versus the algebraic shortcut.
archive = np.load(ROOT/'results/landscape_instances.npz')
U = archive['simplex']; n = U.shape[1]
assert np.max(abs(U.sum(axis=0))) < 1e-12
assert np.max(abs(U.T@U-(n+1)/n*np.eye(n))) < 1e-12
max_coefficient_error = 0.; max_gap_error = 0.; oracle_cases = 0
for j in [0,7,19,39]:
    W,p,c = [archive[f'{j}_{k}'] for k in ['W','p','c']]
    for L0 in [0.,1e-4,.1,1.,100.]:
        b=c-L0*W@(W.T@p)
        xs,fmin,nu=e.constrained_minimum(b,W,L0)
        # KKT stationarity is checked inside constrained_minimum.
        for delta in [1e-5,.003,.05]:
            r=e.R if L0==0 else min(e.R,2*np.sqrt(delta/L0))
            signs=np.sign(U@rng.standard_normal(n)); signs[signs==0]=1
            oracle=np.array([true_value(r*u,W,p,c,L0) for u in U])+delta*signs
            actual=n/(r*(n+1))*(U.T@oracle)
            bias=n/(2*(n+1))*L0*(np.sum((U@W)**2,axis=1)@U)
            v=n/(n+1)*(U.T@signs)
            analytic=b+r*bias+delta/r*v
            cerr=np.linalg.norm(actual-analytic)
            assert cerr < 2e-9*max(1.,np.linalg.norm(analytic))
            x=project(actual)
            g=true_value(x,W,p,c,L0)-true_value(xs,W,p,c,L0)
            gs=e.direct_gap(delta,b,bias,v,W,L0,xs,nu)
            gerr=abs(g-gs)
            assert gerr < 2e-9*max(1.,abs(g))
            max_coefficient_error=max(max_coefficient_error,float(cerr))
            max_gap_error=max(max_gap_error,float(gerr)); oracle_cases+=1
summary.update(oracle_cases=oracle_cases,max_coefficient_error=max_coefficient_error,max_gap_error=max_gap_error)

# Balanced signs at odd dimension attain the endpoint norm bound exactly.
n0=7; S=e.simplex(n0); delta=.01
signs=np.r_[np.ones(4),-np.ones(4)]
chat=n0/(e.R*(n0+1))*delta*(S.T@signs)
gap=.5*e.MU*np.sum(project(chat)**2)
target=n0*delta**2/(2*e.MU*e.R**2)
assert np.isclose(gap,target,rtol=2e-13,atol=1e-15)
summary['endpoint_sharp_constant_test']=True

# Random low-dimensional problems exercise both radius and projection branches.
# A dense grid is evaluated through direct vector computations, without the
# squared polynomial equation. It also detects early failure before a root.
cases=0; finite=0; initial_failure=0; max_crossing_residual=0.
for j in range(240):
    n0=int(rng.integers(2,10)); S=e.simplex(n0)
    k=int(rng.integers(1,n0)); W,_=np.linalg.qr(rng.normal(size=(n0,k)))
    L0=0. if j%12==0 else 10**rng.uniform(-4,2)
    b=rng.normal(size=n0)*10**rng.uniform(-2,.7)
    xs,fmin,nu=e.constrained_minimum(b,W,L0)
    bias=n0/(2*(n0+1))*L0*(np.sum((S@W)**2,axis=1)@S)
    signs=np.sign(S@rng.normal(size=n0)); signs[signs==0]=1
    v=n0/(n0+1)*(S.T@signs)
    t=e.threshold(b,bias,v,W,L0,fmin)
    grid=np.r_[0.,np.geomspace(1e-12,1e5,650)]
    losses=np.array([e.direct_gap(d,b,bias,v,W,L0,xs,nu) for d in grid])
    assert np.all(losses[grid<t*(1-1e-6)]<=e.EPS+3e-8),(j,t,'early failure')
    failed=grid[losses>e.EPS+3e-8]
    if len(failed):assert t<=failed[0]*(1+1e-6)+1e-10,(j,t,failed[0])
    if np.isfinite(t) and t>0:
        residual=abs(e.direct_gap(t,b,bias,v,W,L0,xs,nu)-e.EPS)
        assert residual<3e-8,(j,t,residual)
        assert e.direct_gap(t*(1+1e-5),b,bias,v,W,L0,xs,nu)>=e.EPS-3e-8
        max_crossing_residual=max(max_crossing_residual,float(residual));finite+=1
    elif t==0:initial_failure+=1
    cases+=1
summary.update(random_branch_cases=cases,positive_crossings=finite,initial_failures=initial_failure,max_crossing_residual=max_crossing_residual)

rows=list(csv.DictReader((ROOT/'results/landscape.csv').open()))
assert len(rows)==39
for row in rows:
    assert float(row['lower'])<=float(row['upper'])
    if row['empirical_expanded']:
        assert float(row['empirical_expanded'])+1e-10>=float(row['lower'])
        assert float(row['empirical_expanded'])<=float(row['empirical_fixed16'])+1e-10
summary['saved_rows']=len(rows)
(ROOT/'results/landscape_validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
