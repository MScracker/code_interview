# -*- coding: utf-8 -*- 
# @since : 2020-03-08 15:02 
# @author : wongleon
import re

A = 0
B = 0
C = 0
D = 0
E = 0
err = 0
private = 0


def is_correct_mask(mask):
    if mask == '255.255.255.255' or mask == '0.0.0.0':
        return False
    mask_array = map(int, mask.split('.'))
    binary_mask = 0
    for i in range(len(mask_array)):
        binary_mask += mask_array[i] * pow(2, 8 * (len(mask_array) - i - 1))

    if (binary_mask - 1) | binary_mask == int('0xFFFFFFFF', 16):
        return True
    return False


while True:
    try:
        ip, mask = map(str, raw_input().split('~'))
        if re.match('0\.\d+\.\d+\.\d+', ip):
            continue
        if re.match('\d+\.\d+\.\d+\.\d+', ip) is None or not is_correct_mask(mask):
            err += 1
        else:
            ip_head = int(ip.split('.')[0])
            ip_second = int(ip.split('.')[1])
            if ip_head > 0 and ip_head <= 126:
                A += 1
                if ip_head == 10:
                    private += 1
            elif ip_head >= 128 and ip_head <= 191:
                B += 1
                if ip_head == 172 and ip_second >= 16 and ip_second <= 31:
                    private += 1
            elif ip_head >= 192 and ip_head <= 223:
                C += 1
                if ip_head == 192 and ip_second == 168:
                    private += 1
            elif ip_head >= 224 and ip_head <= 239:
                D += 1
            elif ip_head >= 240 and ip_head <= 255:
                E += 1
    except:
        break
print '{} {} {} {} {} {} {}'.format(A, B, C, D, E, err, private)
