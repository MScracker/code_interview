# -*- coding: utf-8 -*- 
# @since : 2020-03-03 11:38 
# @author : wongleon

try:
    while True:
        line = raw_input()
        if line == '':
            break
        lines = line.split()
        print sum([int(i) for i in lines])
except:
    pass
