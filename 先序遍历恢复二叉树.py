# -*- coding: utf-8 -*- 
# @since : 2020/5/19 11:40 
# @author : wongleon
from common_data_structure import *
from collections import namedtuple, deque


def recoverFromPreorder(S):
    ans = {-1: TreeNode(0)}  # 字典初始化

    '''
    v: values
    p: the depth of current node 
    '''
    def addTree(v, p):  # 添加树函数
        print ans
        ans[p] = TreeNode(int(v))
        if not ans[p - 1].left:  # 左子树不存在就加在左边
            ans[p - 1].left = ans[p]
        else:  # 反之加在右边
            ans[p - 1].right = ans[p]

    val, dep = '', 0  # 值和对应深度初始化
    for c in S:
        if c != '-':
            val += c  # 累加字符来获得数字
        elif val:  # 如果是‘-’且存在val
            addTree(val, dep)  # 就把累加好的数字和对应深度添加进树
            val, dep = '', 1  # 值和对应深度重新初始化
        else:
            dep += 1  # 连续的‘-’只加深度不加值
    addTree(val, dep)  # 末尾剩余的数字也要加进树
    return ans[0]

S = "1-401--349---90--88"
tree = recoverFromPreorder(S)
# values = [1, 2, 2, None, 3, None, 3]
# tree = build_tree_from_list(values)
pretty_print(tree)
