# -*- coding: utf-8 -*- 
# @since : 2020/4/21 21:28 
# @author : wongleon
from common_data_structure import ListNode


class ChainList:
    def __init__(self):
        self.head = None
        self.length = 0

    def is_empty(self):
        '''
        :return: if the link list is empty
        '''
        return self.length == 0

    def length(self):
        '''
        :return: the length of linklist
        '''
        l = 0
        p = self.head
        while p:
            l += 1
            p = p.next
        return l

    def travel(self, index):
        '''
        travel to specific node
        :param index:int
        :return: LinkNode current node
        '''
        p = self.head
        i = 0
        while p:
            if i == index:
                return p
            p = p.next
            i += 1

    def append(self, dataOrNode):
        '''
        :param dataOrNode: int or LinkNode
        :return: None
        '''
        if isinstance(dataOrNode, ListNode):
            item = dataOrNode
        else:
            item = ListNode(dataOrNode)

        if self.head is None:
            self.head = item
            self.length += 1
            return

        tail = self.head
        while tail.next:
            tail = tail.next
        tail.next = item
        self.length += 1

    def insert(self, index, dataOrNode):
        '''
        :param index: start from zero
        :param dataOrNode: int or LinkNode
        :return: None
        '''
        if isinstance(dataOrNode, ListNode):
            item = dataOrNode
        else:
            item = ListNode(dataOrNode)

        if self.head is None:
            self.head = item
            self.length += 1

        if index < 0 or index > self.length:
            raise Exception('index is out of range.')

        if index == 0:
            item.next = self.head
            self.head = item
            self.length += 1
            return

        i = 0
        prev = self.head
        cur = self.head
        while cur and i < index:
            i += 1
            prev = cur
            cur = cur.next
        if i == index:
            prev.next = item
            item.next = cur
            self.length += 1
            return

    def delete(self, index):
        '''
        :param index: start from zero
        :return: None
        '''
        if self.head is None:
            raise Exception('head is None.')

        if index < 0 or index >= self.length:
            raise Exception('index is out of range.')

        if index == 0:
            self.head = self.head.next
            self.length -= 1
            return

        prev = self.head
        cur = self.head
        i = 0
        while cur and i < index:
            i += 1
            prev = cur
            cur = cur.next

        if i == index:
            prev.next = cur.next
            cur.next = None
            self.length -= 1

    def deleteNode(self, node):
        '''
        :param node: LinkNode
        :return: void
        '''
        node.val, node.next = node.next.val, node.next.next

    def search(self, data):
        '''
        :param data: int
        :return: bool, True if Found
        '''
        if self.head is None:
            raise Exception('head is None.')
        cur = self.head
        while cur:
            if cur.val == data:
                return True
            cur = cur.next
        return False

    def update(self, index, data):
        '''
        :param index: int, start from zero
        :param data: int
        :return: None
        '''
        if self.head is None:
            raise Exception('head is None.')
        if index < 0 or index >= self.length:
            raise Exception('index is out of range.')
        cur = self.head
        i = 0
        while cur:
            if i == index:
                cur.val = data
            i += 1
            cur = cur.next

    def clear(self):
        '''
        :return: None
        '''
        if self.head is None:
            raise Exception('head is None.')
        self.head = None
        self.length = 0

    def __repr__(self):
        if self.is_empty():
            return ''
        s = ''
        cur = self.head
        while cur:
            s += str(cur.val) + ' '
            cur = cur.next
        return s.lstrip()

def __main__():
    links = ChainList()

    for i in range(5):
        links.append(i)
    print links
    # print links.delete(3)
    # print links
    # print links.length

    snode = links.travel(3)
    print snode
    print links
    links.deleteNode(snode)
    print links
