# coding: utf-8
import sys

# 以下是主流程
while True:
    try:
        word = raw_input()
        d = dict()
        mi = sys.maxint
        for i in xrange(len(word)):
            if not word[i] not in d:
                d[word[i]] = word.count(word[i])
            if mi > word.count(word[i]):
                mi = word.count(word[i])
        for k, v in d.iteritems():
            if v == mi:
                word = word.replace(k, '')
        print word
    except:
        break
