# -*- coding: utf-8 -*- 
# @since : 2020-03-08 17:21 
# @author : wongleon
import os

i = 0
record_list = []
count_dict = {}
while True:
    try:
        path, rownum = map(str, raw_input().replace('\\', '/').split())
        filename = os.path.basename(path)
        key = filename[-16:] + " " + rownum
        if key in record_list:
            count_dict[key] += 1
        else:
            record_list.append(key)
            count_dict[key] = 1
    except:
        break

for e in record_list[-8:]:
    print "{} {}".format(e, count_dict[e])
