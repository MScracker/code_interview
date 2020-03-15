# -*- coding: utf-8 -*- 
# @since : 2020-03-03 20:46 
# @author : wongleon

from scipy.optimize import fsolve

def pmix(x):
    return x

def paera(x, y):
    return x**2 - y**2

def findIntersection(fun1, fun2, x0):
    return [fsolve(lambda x:fun1(x)-fun2(x, y), x0) for y in range(1, 10)]

print findIntersection(pmix, paera, 0)