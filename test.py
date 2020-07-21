# -*- coding: utf-8 -*- 
# @since : 2020/6/5 17:07 
# @author : wongleon
from __future__ import print_function
from collections import *

a = ['a', 'a', 'b']
P = namedtuple('Point', 'x, y, z', True)
p = P(1,2,3)
print(p.x, p.y, p.z, sep= ',')