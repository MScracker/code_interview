# -*- coding: utf-8 -*- 
# @since : 2020/6/1 16:11 
# @author : wongleon

from common_data_structure import *
from tree_traverse import *


def mergeTrees(t1, t2):
    """
    :type t1: TreeNode
    :type t2: TreeNode
    :rtype: TreeNode
    """

    def helper(t1, t2):
        if not (t1 and t2):
            return t1 if t1 else t2
        t1.val += t2.val
        t1.left = helper(t1.left, t2.left)
        t1.right = helper(t1.right, t2.right)
        return t1

    return helper(t1, t2)


t1 = build_tree_from_list(range(10))
# t2 = build_tree_from_list([2, 1, 3, None, 4, None, 7])

# mergeTrees(t1, t2)
# print level_traverse_bfs(t1)
def invertTree(root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """

    def helper(root):
        if not root:
            return

        right = helper(root.right)
        left = helper(root.left)
        root.left = right
        root.right = left
        return root

    return helper(root)

pretty_print(t1)
invertTree(t1)
pretty_print(t1)