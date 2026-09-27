"""Theorem-based bounds used by exp_landscape.py.
"""
import numpy as np
from decimal import Decimal, getcontext
class HighPrecision:
    mpf = staticmethod(lambda x: Decimal(str(x)))
    sqrt = staticmethod(lambda x: Decimal(x).sqrt())
    log = staticmethod(lambda x: Decimal(x).ln())
    @staticmethod
    def asin(x):
        # asin power series; |x| <= 1/4 here, fast absolute convergence.
        term = total = x
        for k in range(200):
            term *= Decimal((2*k+1)**2)*x*x / Decimal(2*(k+1)*(2*k+3))
            total += term
            if abs(term) < Decimal('1e-75'): return total
        raise ArithmeticError('asin series failed to converge')
mp=HighPrecision()
getcontext().prec = 70
n, mu, R, eps, T, beta = 256, 1., 1., .01, 257, .25
th_exact = mp.sqrt(2*mp.log(2*mp.mpf(T+1)/(1-mp.mpf('0.25')))/(n-1))
th = np.nextafter(float(th_exact), np.inf)
D = lambda x: mp.mpf(format(float(x), '.17g'))
thh = D(th)
assert thh <= mp.mpf('.25') and 2*(T+1)*(1-thh**2)**(mp.mpf(n-1)/2) < 1-D(beta)
kappa, t, q = np.meshgrid(np.geomspace(1.2,60,60),np.geomspace(.001,mu*R**2/eps,60),np.linspace(.02,1,60),indexing='ij')
rho=np.sqrt(t*eps/mu); a=kappa*eps/rho; m=a/q
r0=rho*(th+q)/(1-q*th)
up_round=lambda x: float(x)*(1+1e-11)


def bracket(L0):
    gap=a*rho-.5*mu*rho**2-a*th*r0-.5*m*m/L0
    bound=np.where(gap>eps,a*(r0+m/L0)*np.arcsin(th),np.inf)
    idx=np.unravel_index(np.argmin(bound),bound.shape)
    endpoint=up_round(D(R)*mp.sqrt(3*D(mu)*D(eps))*thh)
    row=dict(L0_over_mu=L0/mu, upper=endpoint, source='quadratic',rho='',a='',m='',separation_margin='', certificate='')
    if np.isfinite(bound[idx]):
        rr,aa,mm=D(rho[idx]),D(a[idx]),D(m[idx]); ll=D(L0)
        rzero=rr*(mm*thh+aa)/(mm-aa*thh)
        margin=aa*rr-D(mu)*rr**2/2-aa*thh*rzero-mm**2/(2*ll)-D(eps)
        assert 0<rr<=D(R) and 0<aa<=mm and margin>0
        cert=up_round(aa*(rzero+mm/ll)*mp.asin(thh))
        row.update(rho=str(rr),a=str(aa),m=str(mm),separation_margin=str(margin),certificate=cert)
        if cert<endpoint: row.update(upper=cert,source='belt')
    # Solve near-endpoint guarantee for delta at fixed r, then maximize the
    # resulting concave quadratic in r. Zero denotes no positive guarantee.
    c=mp.sqrt(2*D(mu)*D(eps))-D(L0)*D(R)
    low=mp.mpf(0); rstar=mp.mpf(0)
    if c>0:
        rstar=min(D(R),2*c/(mp.sqrt(n)*D(L0)))
        low=c*rstar/mp.sqrt(n)-D(L0)*rstar**2/4
    row.update(lower=float(low)*(1-1e-11),query_radius=float(rstar))
    if low>0:
        err=mp.sqrt(n)*(D(L0)*rstar/4+D(row['lower'])/rstar)+D(L0)*D(R)
        assert err**2/(2*D(mu)) <= D(eps)
    return row
