# -*- coding: utf-8 -*- 
# @since : 2020-03-08 18:22 
# @author : wongleon
import re


def is_valid(pw):
    cnt = 0
    if re.search('[A-Z]', pw):
        cnt += 1
    if re.search('[0-9]', pw):
        cnt += 1
    if re.search('[a-z]', pw):
        cnt += 1

    if re.search('[^a-zA-Z0-9]', pw):
        cnt += 1
    if cnt >= 3:
        return True
    return False


def no_repeate_substring(pw):
    for i in range(len(pw) - 3):
        if pw.count(pw[i: i + 3]) > 1:
            return False
    return True


while True:
    try:
        pw = raw_input()
        if len(pw) > 8 and is_valid(pw) and no_repeate_substring(pw):
            print('OK')
        else:
            print('NG')
    except:
        break
# 021Abc9000
# 021Abc9Abc1
# 021ABC9000
# 021$bc9000