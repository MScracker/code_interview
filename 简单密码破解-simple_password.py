# -*- coding: utf-8 -*- 
# @since : 2020-03-09 11:49 
# @author : wongleon
while True:
    try:
        clear_text = raw_input()
        secret_text = ''
        for i in range(len(clear_text)):
            if clear_text[i].isupper():
                t = ord(clear_text[i]) - ord('A') + ord('b')
                if (t > ord('z')):
                    t = t - ord('z') - 1 + ord('a')
                secret_text += chr(t)
            elif clear_text[i].islower():
                if clear_text[i] in 'abc':
                    secret_text += '2'
                elif clear_text[i] in 'def':
                    secret_text += '3'
                elif clear_text[i] in 'ghi':
                    secret_text += '4'
                elif clear_text[i] in 'jkl':
                    secret_text += '5'
                elif clear_text[i] in 'mno':
                    secret_text += '6'
                elif clear_text[i] in 'pqrs':
                    secret_text += '7'
                elif clear_text[i] in 'tuv':
                    secret_text += '8'
                elif clear_text[i] in 'wxyz':
                    secret_text += '9'
                else:
                    secret_text += clear_text[i]
            else:
                secret_text += clear_text[i]

        print secret_text
    except:
        break
