import numpy as np
from matplotlib import pyplot as plt
import _lib.pr_func as pr
from _lib.utility import *

## %% serial case

N = 20
M = 10
K = 20
pr.set_dims([('w',N),('x',M),('a',K)])
U = pr.func(val=gauss_utility(N,K,sigma=2),vars=['w','a'])

## %%

beta1 = 100.0
beta2 = 100.0

pw = pr.func(vars=['w'], val='unif').normalize()
px = pr.func(vars=['x'], val='unif').normalize()
pa = pr.func(vars=['a'], val='unif').normalize()
F = pr.func('f(w,x)', val='rnd')
pxgw_temp = 0

for i in range(0,10000):
    pxgw = (px*pr.exp(beta1*F)).normalize(['x'])
    px = pr.sum(pw*pxgw,['w'])
    pwgx = (pxgw*pw)/px
    pagx = (pa*pr.exp(beta2*pr.sum(pwgx*U,['w']))).normalize(['a'])
    pa = pr.sum(pxgw*pw*pagx,['w','x'])
    F = pr.sum(pagx*(U-pr.log(pagx/pa)/beta2),['a'])

    if np.linalg.norm(pxgw.val-pxgw_temp)<1e-10: break
    pxgw_temp = pxgw.val

pagw = pr.sum(pxgw*pagx,['x'])
plt.pcolor(pagw.val)
plt.show()



## %% 2d hidden variable case

N = 20
M1 = 10
M2 = 10
K = 20
pr.set_dims([('w',N),('x1',M1),('x2',M2),('a',K)])
U = pr.func(val=utility(N,K,sigma=2),vars=['w','a'])

## %%

beta1 = 80.0
beta2 = 90.0
gamma = 100.0

pw = pr.func(vars=['w'], val='unif').normalize()
px1 = pr.func(vars=['x1'], val='unif').normalize()
px2 = pr.func(vars=['x2'], val='unif').normalize()
pa = pr.func(vars=['a'], val='unif').normalize()
pagx1x2 = pr.func(vars=['a','x1','x2'], val='rnd').normalize(['a'])
F1 = pr.func('f(w,x1)', val='rnd')
pagx1x2_temp = 0

for i in range(0,10000):
    px1gw = (px1*pr.exp_tr(beta1*F1)).normalize(['x1'])
    px1 = pr.sum(pw*px1gw,['w'])
    F2 = pr.sum(px1gw*pagx1x2*(U-pr.log(pagx1x2/pa)/gamma),['a','x1'])
    px2gw = (px2*pr.exp_tr(beta2*F2)).normalize(['x2'])
    px2 = pr.sum(pw*px2gw,['w'])
    pwgx1x2 = (px1gw*px2gw*pw).normalize(['w'])
    pagx1x2 = (pa*pr.exp_tr(gamma*pr.sum(pwgx1x2*U,['w']))).normalize(['a'])
    pa = pr.sum(px1gw*px2gw*pw*pagx1x2,['w','x1','x2'])
    F1 = pr.sum(px2gw*pagx1x2*(U-pr.log(pagx1x2/pa)/gamma),['a','x2'])

    if np.linalg.norm(pagx1x2.val-pagx1x2_temp)<1e-10: break
    pagx1x2_temp = pagx1x2.val

pagw = pr.sum(pagx1x2*px1gw*px2gw,['x1','x2'])

plt.pcolor(pagw)
plt.show()
