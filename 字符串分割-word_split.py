# -*- coding: utf-8 -*- 
# @since : 2020-03-03 13:39 
# @author : wongleon

while True:
    try:
        str = raw_input()
        for i in range(len(str) / 8):
            print str[i * 8 : i * 8 + 8]
        if len(str) % 8 != 0:
            last = len(str) / 8 * 8
            print str[-(len(str) % 8):] + '0' * (8 - len(str) % 8)
    except:
        break