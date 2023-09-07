#!/usr/bin/env python
# coding=utf-8
class LinkNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class LinkedList:

    def __init__(self):
        self.dummy_node = LinkNode(0)
        self.num_nodes = 0

    def get(self, index: int) -> int:
        if index >= self.num_nodes:
            return -1

        pcur = self.dummy_node.next  # start from the true head
        for _ in range(index):
            pcur = pcur.next

        return pcur.val

    def addAtHead(self, val: int) -> None:
        # self.num_nodes += 1
        # new_node = LinkNode(val)
        # new_node.next = self.dummy_node.next
        # self.dummy_node.next = new_node
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        # self.num_nodes += 1
        # new_node = LinkNode(val)
        # pcur = self.dummy_node
        # while pcur.next:
        #     pcur = pcur.next
        # pcur.next = new_node
        self.addAtIndex(self.num_nodes, val)

    def addAtIndex(self, index: int, val: int) -> None:
        new_node = LinkNode(val)
        if index > self.num_nodes:
            return

        pcur = self.dummy_node
        for _ in range(index):
            pcur = pcur.next
        self.num_nodes += 1
        new_node.next = pcur.next
        pcur.next = new_node

    def deleteAtIndex(self, index: int) -> None:
        if index >= self.num_nodes:
            return

        pcur = self.dummy_node
        for _ in range(index):
            pcur = pcur.next

        self.num_nodes -= 1
        pcur.next = pcur.next.next

        # i = 0
        # while pcur.next:
        #     if i == index:
        #         self.num_nodes -= 1
        #         pcur.next = pcur.next.next
        #     else:
        #         pcur = pcur.next

        #     i += 1


# Your MyLinkedList object will be instantiated and called as such:
myLinkedList = LinkedList();
myLinkedList.addAtHead(1);
myLinkedList.addAtTail(3);
myLinkedList.addAtIndex(1, 2);  # 链表变为 1->2->3
res = myLinkedList.get(1);  # 返回 2
print(res)
myLinkedList.deleteAtIndex(1);  # 现在，链表变为 1->3
res1 = myLinkedList.get(1);  # 返回 3
print(res1)
