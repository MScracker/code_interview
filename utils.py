#!/usr/bin/env python
# coding=utf-8
from common_data_structure import TreeNode
from typing import List, Optional
import math


def getTreeDepth(root: TreeNode):
    if not root:
        return 0
    return 1 + max(getTreeDepth(root.left), getTreeDepth(root.right))


def printTree(root: TreeNode):
    maxLevel = getTreeDepth(root)
    printNodeInternal([root], 1, maxLevel)


def printNodeInternal(nodes: List[TreeNode], level: int, maxLevel: int):
    if not nodes or isAllElementsNull(nodes):
        return

    floor = maxLevel - level
    endgeLines = int(math.pow(2, max(floor - 1, 0)))
    firstSpaces = int(math.pow(2, floor) - 1)
    betweenSpaces = int(math.pow(2, floor + 1) - 1)

    printWhitespaces(firstSpaces)

    newNodes = []
    for node in nodes:
        if node:
            print(node.val, end="")
            newNodes.append(node.left)
            newNodes.append(node.right)
        else:
            newNodes.append(None)
            newNodes.append(None)
            print(" ", end="")

        printWhitespaces(betweenSpaces)

    print("")

    for i in range(1, endgeLines + 1):
        for j in range(len(nodes)):
            printWhitespaces(firstSpaces - i)
            if not nodes[j]:
                printWhitespaces(endgeLines + endgeLines + i + 1)
                continue

            if nodes[j].left:
                print("/", end="")
            else:
                printWhitespaces(1)

            printWhitespaces(i + i - 1)
            if nodes[j].right:
                print("\\", end="")
            else:
                printWhitespaces(1)
            printWhitespaces(endgeLines + endgeLines - i)

        print("")

    printNodeInternal(newNodes, level + 1, maxLevel)


def printWhitespaces(count: int):
    for i in range(count):
        print(" ", end="")


def isAllElementsNull(list: List[TreeNode]):
    for object in list:
        if object:
            return False
    return True
