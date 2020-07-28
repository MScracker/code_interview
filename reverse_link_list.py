# -*- coding: utf-8 -*- 
# @since : 2020/5/18 17:06 
# @author : wongleon
from ChainList import *

chain_list = ChainList()
for i in range(5):
    chain_list.append(i)


def reverseLinkList(head):
    if head == None or head.next == None:
        return head
    tail = reverseLinkList(head.next) #获取尾节点
    head.next.next = head #head为尾节点, 指向前一节点
    head.next = None #删除原正序连接
    return tail


def recursiveReverseLinkList(head):
    prev = None
    cur = head
    while cur:
        next_node = cur.next
        cur.next = prev
        prev = cur
        cur = next_node
    return prev


x = recursiveReverseLinkList(chain_list.head)

print x
