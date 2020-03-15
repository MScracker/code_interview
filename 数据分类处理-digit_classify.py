# -*- coding: utf-8 -*- 
# @since : 2020-03-12 17:22 
# @author : wongleon
while True:
    res = []
    try:
        r1 = raw_input().split(' ', 1)
        a, I = r1[0], r1[1]
        r2 = raw_input().split(' ', 1)
        b, R = r2[0], r2[1]
        I = I.split()
        R = map(str, sorted(list(set(map(int, R.split())))))
        d = {}
        for i in range(len(I)):
            for e in R:
                if I[i].count(e) < 1:
                    continue
                else:
                    if e in d:
                        d[e].append(i)
                    else:
                        d[e] = [i]
        for e in map(str, sorted(map(int, d.keys()))):
            res.append(e)
            res.append(len(d[e]))
            for idx in d[e]:
                res.append(idx)
                res.append(I[idx])
        res.insert(0, len(res))
        print ' '.join(map(str, res))
    except:
        break
