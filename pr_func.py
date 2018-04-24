# -*- coding: utf-8 -*-
# pr_func.py
# implementation of a 'function' container class containing a numpy
# array and knowledge about its dependent variables.
# Sebastian Gottwald <sebastian.gottwald@uni-ulm.de> 2018

import numpy as np
from collections import OrderedDict

global dims

def set_dims(d=[]):
    global dims
    dims = OrderedDict(d)

def get_r(variables,dims):
    r = []
    for i in range(0,len(dims)):
        for item in variables:
            if dims.items()[i][0] == item:
                r.append(i)
    return r

def exp(fnc):
    return func(val=np.exp(fnc.val), r=fnc.r, parse_name=False)

def exp_tr(fnc):
    return func(val=np.exp(fnc.val-np.max(fnc.val)), r=fnc.r, parse_name=False)

def log(fnc):
    return func(val=np.log(fnc.val+1e-55), r=fnc.r, parse_name=False)

def sum(fnc,over=[]):
    r_over = get_r(over,dims)
    r = [item for item in fnc.r if not(item in r_over)]
    val = np.einsum(fnc.val,fnc.r,r)
    return func(val=val,r=r,parse_name=False)


class func(object):
    """function class that knows about its variables"""

    def __init__(self,name='',val='',vars=[],r=[],parse_name=True):
        # set vars:
        if len(vars)>0:
            self.vars = vars
        elif parse_name:
            self.vars = self.parse_name(name)
        elif len(r)>0:
            self.vars = [dims.items()[ind][0] for ind in r]
        else:
            self.vars = []
        # set r:
        if len(r) >0:
            self.r = r
        else:
            self.r = get_r(self.vars,dims)
        # set dims:
        self.dims = [dims.items()[ind][1] for ind in self.r]
        # set val:
        if type(val) == str:
            if val == 'rnd':
                self.val = np.random.rand(*self.dims)
            elif val == 'unif':
                self.val = np.ones(self.dims)
            else:
                self.val = 0
        else:
            self.val = val

    def parse_name(self,name):
        if name.count('f') == 1:
            vars = name[2:-1].split(',')
        return vars

    def get_name(self):
        return "f({})".format(','.join(self.vars))

    def __str__(self):
        return self.get_name()

    def __mul__(self,other):
        if type(other)==int or type(other)==float:
            return func(val=other*self.val,vars=self.vars,r=self.r,parse_name=False)
        else:
            vars = list(set(self.vars + other.vars))
            r = sorted(list(set(self.r+other.r)))
            result = np.einsum(other.val,other.r,self.val,self.r,r)
            return func(val=result,vars=vars,r=r,parse_name=False)

    def __rmul__(self,other):
        if type(other)==int or type(other)==float:
            return func(val=other*self.val,vars=self.vars,r=self.r,parse_name=False)

    def __div__(self,other):
        if type(other)==int or type(other)==float:
            return func(val=self.val/float(other),vars=self.vars,r=self.r,parse_name=False)
        else:
            r = sorted(list(set(self.r+other.r)))
            vars = list(set(self.vars + other.vars))
            result = np.einsum(1.0/(other.val+1e-55),other.r,self.val,self.r,r)
            return func(val=result,vars=vars,r=r,parse_name=False)

    def __rdiv__(self,other):
        if type(other)==int or type(other)==float:
            return func(val=float(other)/self.val,vars=self.vars,r=self.r,parse_name=False)

    def __pos__(self):
        return self

    def __neg__(self):
        return (-1)*self

    def __add__(self,other):
        if type(other)==int or type(other)==float:
            return func(val=self.val+other,vars=self.vars,r=self.r,parse_name=False)
        else:
            vars = list(set(self.vars+other.vars))
            r = sorted(list(set(self.r+other.r)))
            I = np.ones([dims.items()[ind][1] for ind in r])
            s0 = np.einsum(I,r,self.val,self.r,r)
            s1 = np.einsum(I,r,other.val,other.r,r)
            return func(val=s0+s1,vars=vars,r=r,parse_name=False)

    def __radd__(self,other):
        if type(other)==int or type(other)==float:
            return func(val=self.val+other,vars=self.vars,r=self.r,parse_name=False)

    def __sub__(self,other):
        return self+(-other)

    def normalize(self,vrs=[]):
        if len(vrs) == 0:
            vrs = self.vars
        r_vrs = get_r(vrs,dims)
        r_Z = [rval for rval in self.r if not(rval in r_vrs)]
        Z = np.einsum(self.val,self.r,r_Z)
        self.val = np.einsum(self.val,self.r,1.0/(Z+1e-55),r_Z,self.r)
        return self
