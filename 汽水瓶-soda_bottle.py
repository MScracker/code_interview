# -*- coding: utf-8 -*- 
# @since : 2020-03-09 12:24 
# @author : wongleon
while True:
    try:
        n = int(raw_input())
        if n == 0:
            break
        answer = 0
        while (n > 1):
            if (n % 3 == 2):
                if (n < 3):
                    answer += 1
                    n = 1
                else:
                    answer += n / 3
                    n = n % 3 + n / 3
            else:
                answer += n / 3
                n = n % 3 + n / 3
        print answer
    except:
        break

