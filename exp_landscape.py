"""Deterministic reanalysis of the paper's smooth strongly convex landscape.
"""
from pathlib import Path
import argparse,csv,json,hashlib,platform,time
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import brentq
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from maln.landscape_bounds import bracket,th

ROOT=Path(__file__).resolve().parent
N,MU,R,EPS,BETA=256,1.,1.,.01,.25
SEED=20260926


def simplex(n):
    E=np.eye(n+1)-1/(n+1)
    Q,_=np.linalg.qr(E[:,:n]);U=E@Q
    return U/np.linalg.norm(U,axis=1,keepdims=True)


def make_pool(n,count,random_directions):
    rng=np.random.default_rng(SEED);pool=[]
    # Preserve the original distribution and original generation order for 40 objectives.
    for j in range(count):
        k=int(rng.integers(1,max(2,n//4)))
        p=rng.standard_normal(n);p*=rng.uniform(.2,1.5)*R/np.linalg.norm(p)
        A=rng.standard_normal((n,k))
        if rng.random()<.5:A[:,0]=p
        W,_=np.linalg.qr(A)
        c=rng.standard_normal(n)
        if rng.random()<.5:c=-rng.uniform(0,1)*p+.3*c/np.linalg.norm(c)
        c*=rng.uniform(0,3)*MU*R/np.linalg.norm(c)
        pool.append(dict(W=W,p=p,c=c))
    for x in pool:x['random']=rng.standard_normal((n,random_directions))
    return pool


def constrained_minimum(b,W,L0):
    bp=W@(W.T@b);bo=b-bp
    p2,o2=bp@bp,bo@bo
    def norm(nu):return np.sqrt(p2/(MU+L0+nu)**2+o2/(MU+nu)**2)
    if norm(0)<=R:nu=0.
    else:
        hi=max(1.,np.linalg.norm(b)/R)
        while norm(hi)>R:hi*=2
        nu=brentq(lambda v:norm(v)-R,0.,hi,xtol=1e-14,rtol=1e-14)
    xs=-bp/(MU+L0+nu)-bo/(MU+nu)
    fmin=.5*MU*(xs@xs)+.5*L0*np.sum((W.T@xs)**2)+b@xs
    kkt=np.linalg.norm(MU*xs+L0*W@(W.T@xs)+b+nu*xs)
    assert kkt<1e-10 and np.linalg.norm(xs)<=R+1e-12
    return xs,fmin,nu


def scalars(c0,d,b,W):
    wp=W.T@c0;wd=W.T@d
    return np.array([c0@c0,2*(c0@d),d@d]),np.array([wp@wp,2*(wp@wd),wd@wd]),np.array([b@c0,b@d])


def gap_scalar(s,B,P,D,L0,fmin):
    b2=max(0.,poly.polyval(s,B));p2=max(0.,poly.polyval(s,P));bd=poly.polyval(s,D)
    norm=np.sqrt(b2)
    alpha=1/MU if norm<=MU*R else R/norm
    return .5*alpha*alpha*(MU*b2+L0*p2)-alpha*bd-fmin


def real_roots(p):
    p=poly.polytrim(p,tol=1e-25)
    if len(p)<2:return []
    p=p/max(abs(p))
    roots=poly.polyroots(p)
    return sorted(float(z.real) for z in roots if abs(z.imag)<=2e-7*max(1,abs(z.real)))


def first_crossing(c0,d,b,W,L0,fmin,lo,hi):
    B,P,D=scalars(c0,d,b,W)
    # Interior projected point: x=-(c0+s*d)/mu.
    interior=poly.polyadd(.5/MU*B,.5*L0/MU**2*P)
    interior=poly.polysub(interior,D/MU);interior[0]-=fmin+EPS
    # Boundary: x=-R(c0+s*d)/||c0+s*d||. Squaring gives candidates only.
    A=(.5*MU*R*R-fmin-EPS)*B+.5*L0*R*R*P
    exterior=poly.polysub(poly.polymul(A,A),R*R*poly.polymul(poly.polymul(D,D),B))
    roots=real_roots(interior)+real_roots(exterior)
    candidates=sorted(set([lo]+[x for x in roots if x>=lo-1e-10 and (not np.isfinite(hi) or x<=hi+1e-10)]+([hi] if np.isfinite(hi) else [])))
    for i,s in enumerate(candidates):
        s=max(lo,s)
        if s>hi:continue
        gap=gap_scalar(s,B,P,D,L0,fmin)
        # Reject extraneous roots introduced by squaring or by using the wrong branch.
        if s!=lo and abs(gap-EPS)>2e-7*max(EPS,abs(fmin),1e-3):continue
        nexts=candidates[i+1] if i+1<len(candidates) else np.inf
        ds=max(1e-8,abs(s)*1e-5)
        if np.isfinite(nexts):ds=min(ds,max(0,(nexts-s)/4))
        if np.isfinite(hi):ds=min(ds,max(0,(hi-s)/4))
        if ds==0:continue
        right=s+ds
        if gap_scalar(right,B,P,D,L0,fmin)>EPS+1e-12:
            if s==lo and gap>EPS+1e-10:return lo
            left=max(lo,s-ds)
            # Refine in the unsquared equation, using a LOCAL crossing bracket.
            fun=lambda v:gap_scalar(v,B,P,D,L0,fmin)-EPS
            if fun(left)<=0 and fun(right)>0:
                root=brentq(fun,left,right,xtol=1e-13,rtol=1e-12)
                return root
    return np.inf


def threshold(b,bias,v,W,L0,fmin):
    if L0==0:return first_crossing(b,v/R,b,W,L0,fmin,0,np.inf)
    # r(delta)=min(R,2 sqrt(delta/L0)); s=sqrt(delta) on the first branch.
    cut=L0*R*R/4;smax=np.sqrt(cut)
    d=2*bias/np.sqrt(L0)+np.sqrt(L0)*v/2
    t=first_crossing(b,d,b,W,L0,fmin,0,smax)
    if np.isfinite(t):return t*t
    return first_crossing(b+R*bias,v/R,b,W,L0,fmin,cut,np.inf)


def direct_gap(delta,b,bias,v,W,L0,xs,nu):
    if L0==0:r=R
    elif delta==0:r=0.
    else:r=min(R,2*np.sqrt(delta/L0))
    chat=b if r==0 else b+r*bias+delta/r*v
    x=-chat/MU
    if np.linalg.norm(x)>R:x*=R/np.linalg.norm(x)
    d=x-xs
    return .5*MU*(d@d)+.5*L0*np.sum((W.T@d)**2)-nu*(xs@d)


def compute(count=40,random_directions=64):
    assert random_directions>=4
    tic=time.time();U=simplex(N);pool=make_pool(N,count,random_directions)
    archive={f'{j}_{key}':val for j,x in enumerate(pool) for key,val in x.items()}
    archive['simplex']=U
    np.savez_compressed(ROOT/'results/landscape_instances.npz',**archive)
    rows=[];witnesses=[];checks=[]
    L0s=np.unique(np.r_[np.logspace(-4,3,36),3.5,3.75,4.])
    for L0 in L0s:
        row=bracket(float(L0));best=np.inf;basebest=np.inf;witness=None
        if L0<=1.00000001:
            for j,obj in enumerate(pool):
                W,p,c=obj['W'],obj['p'],obj['c'];b=c-L0*W@(W.T@p)
                xs,fmin,nu=constrained_minimum(b,W,L0)
                uq=U@W
                bias=N/(2*(N+1))*L0*(np.sum(uq*uq,axis=1)@U)
                dirs=np.column_stack([xs,c,W[:,0],p,obj['random']])
                signs=np.sign(U@dirs);signs[signs==0]=1
                V=N/(N+1)*(U.T@signs)
                V=np.column_stack([V,-V])
                baseids=set(list(range(8))+list(range(dirs.shape[1],dirs.shape[1]+8)))
                for k in range(V.shape[1]):
                    v=V[:,k]
                    t=threshold(b,bias,v,W,L0,fmin)
                    if k in baseids:basebest=min(basebest,t)
                    if t<best:
                        best=t;witness=dict(L0=float(L0),instance=j,pattern=k,delta=float(t))
                # Independent deterministic grid check for a fixed subset of patterns.
                for k in [0,4,7,dirs.shape[1]+1,V.shape[1]-1]:
                    t=threshold(b,bias,V[:,k],W,L0,fmin)
                    if np.isfinite(t) and t>0:
                        before=direct_gap(t*(1-1e-5),b,bias,V[:,k],W,L0,xs,nu)
                        after=direct_gap(t*(1+1e-5),b,bias,V[:,k],W,L0,xs,nu)
                        assert before<=EPS+2e-9 and after>=EPS-2e-9,(L0,j,k,t,before,after)
                        grid=np.geomspace(max(t*1e-9,1e-14),t*(1-1e-5),40)
                        mx=max(direct_gap(d,b,bias,V[:,k],W,L0,xs,nu) for d in grid)
                        assert mx<=EPS+2e-9,(L0,j,k,t,mx)
                        checks.append([float(L0),j,k,float(t),float(before),float(after)])
            assert best+1e-9>=row['lower'],(L0,best,row['lower'])
            assert best<=basebest+1e-10
            if witness:
                j=witness['instance'];k=witness['pattern'];obj=pool[j];W,p,c=obj['W'],obj['p'],obj['c'];b=c-L0*W@(W.T@p)
                xs,fmin,nu=constrained_minimum(b,W,L0);uq=U@W;bias=N/(2*(N+1))*L0*(np.sum(uq*uq,axis=1)@U)
                dirs=np.column_stack([xs,c,W[:,0],p,obj['random']]);sgn=np.sign(U@dirs);sgn[sgn==0]=1
                V=N/(N+1)*(U.T@sgn);V=np.column_stack([V,-V]);v=V[:,k]
                witness.update(gap_at_crossing=direct_gap(best,b,bias,v,W,L0,xs,nu),gap_before=direct_gap(best*(1-1e-5),b,bias,v,W,L0,xs,nu),gap_after=direct_gap(best*(1+1e-5),b,bias,v,W,L0,xs,nu))
                witnesses.append(witness)
        row.update(empirical_fixed16=float(basebest) if np.isfinite(basebest) else '',empirical_expanded=float(best) if np.isfinite(best) else '')
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['L0_over_mu','lower','upper','empirical_fixed16','empirical_expanded']}),flush=True)
    with (ROOT/'results/landscape.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    def at(x):return min(rows,key=lambda r:abs(r['L0_over_mu']-x))
    meta=dict(n=N,mu=MU,R=R,eps=EPS,beta=BETA,T=N+1,theta=th,seed=SEED,instances=count,structured_directions=4,random_directions=random_directions,
              patterns_per_instance=2*(4+random_directions),baseline_patterns=16,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
              upstream_commit='6f235aff2185967980a08166df26679783338bf4',elapsed_seconds=time.time()-tic,independent_checks=len(checks),
              ratio_at_point1=at(.1)['empirical_expanded']/at(.1)['lower'],ratio_fixed16_at_point1=at(.1)['empirical_fixed16']/at(.1)['lower'],
              endpoint_normalized=R*np.sqrt(2*MU*EPS/N)/EPS,first=rows[0],
              interpretation='First failure on the frozen finite family of full-amplitude perturbation patterns along the stated radius rule; neither a uniform guarantee nor the true worst-case threshold.',
              instance_archive_sha256=hashlib.sha256((ROOT/'results/landscape_instances.npz').read_bytes()).hexdigest())
    (ROOT/'results/landscape.json').write_text(json.dumps(meta,indent=2))
    (ROOT/'results/landscape_witnesses.json').write_text(json.dumps(witnesses,indent=2))
    np.savetxt(ROOT/'results/landscape_crossing_checks.csv',np.array(checks),delimiter=',',header='L0,instance,pattern,threshold,gap_before,gap_after',comments='')
    return meta


def plot():
    rows=list(csv.DictReader((ROOT/'results/landscape.csv').open()))
    meta=json.loads((ROOT/'results/landscape.json').read_text())
    x=np.array([float(r['L0_over_mu']) for r in rows]);up=np.array([float(r['upper']) for r in rows])/EPS;lo=np.array([float(r['lower']) for r in rows])/EPS
    mask=np.array([bool(r['empirical_expanded']) for r in rows]);eb=np.array([float(r['empirical_fixed16'] or 0) for r in rows])/EPS;ee=np.array([float(r['empirical_expanded'] or 0) for r in rows])/EPS
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.55,3.25),layout='constrained')
    ax.fill_between(x,up,30,color='#c47429',alpha=.08,lw=0);ok=lo>0;ax.fill_between(x[ok],1e-3,lo[ok],color='#246595',alpha=.08,lw=0)
    ax.loglog(x,up,color='#bd671f',lw=1.5,label='Upper bound on MALN')
    ax.loglog(x[ok],lo[ok],color='#246595',lw=1.5,label='Lower bound: simplex guarantee')
    em=mask&(ee>0);bm=mask&(eb>0)
    ax.loglog(x[bm],eb[bm],'+',color='#536779',ms=2.8,mew=.6,zorder=5,label=f"First failure: {meta['baseline_patterns']} fixed patterns")
    ax.loglog(x[em],ee[em],'o',mfc='white',mec='#146b65',ms=3.7,mew=.8,zorder=4,label=f"First failure: {meta['patterns_per_instance']} fixed patterns")
    ax.set(xlabel=r'$(L-\mu)/\mu$',ylabel=r'Noise level $\delta/\varepsilon$',xlim=(x.min(),x.max()),ylim=(1e-3,30))
    ax.text(1.5e-4,12,'Inadmissible',fontsize=7,color='#925016');ax.text(1.5e-4,.16,'Admissible',fontsize=7,color='#246595')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),borderaxespad=0,frameon=False,fontsize=6.8,handlelength=1.5)
    fig.savefig(ROOT/'figures/landscape.pdf');fig.savefig(ROOT/'figures/landscape.png',dpi=240)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',nargs='?',choices=['plot'],help='legacy alias for --plot-only')
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--instances',type=int,default=40)
    parser.add_argument('--random-directions',type=int,default=64)
    args=parser.parse_args()
    (ROOT/'results').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    if not (args.plot_only or args.mode=='plot'):print(json.dumps(compute(args.instances,args.random_directions),indent=2))
    plot()
