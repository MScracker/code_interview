# -*- coding: utf-8 -*- 
# @since : 2020-03-09 15:18 
# @author : wongleon
import re

while True:
    try:
        word = raw_input()
        alphbet = re.findall(r'[a-zA-Z]', word)
        alphbet.sort(key=lambda a: a.lower())
        nonalphbet = re.findall(r'[^a-zA-Z]', word)
        j = 0
        word_list = list(word)
        for i in range(len(word_list)):
            if word_list[i] in nonalphbet:
                continue
            else:
                word_list[i] = alphbet[j]
                j += 1

        print ''.join(word_list)
    except:
        break
