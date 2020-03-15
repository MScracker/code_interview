# -*- coding: utf-8 -*- 
# @since : 2020-03-08 14:00 
# @author : wongleon
import re

lines = raw_input().split(';')
coordinates = []
for e in lines:
    if re.match('^[A-Z]\d{1,2}$', e):
        coordinates.append(e)

destination = [0, 0]
for c in coordinates:
    if c.startswith('A'):
        destination[0] -= int(c[1:])
    elif c.startswith('W'):
        destination[1] += int(c[1:])
    elif c.startswith('D'):
        destination[0] += int(c[1:])
    elif c.startswith('S'):
        destination[1] -= int(c[1:])

print '{},{}'.format(destination[0], destination[1])