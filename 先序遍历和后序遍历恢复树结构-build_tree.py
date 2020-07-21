# -*- coding: utf-8 -*- 
# @since : 2020/5/19 17:20 
# @author : wongleon

from common_data_structure import *


def buildTree(preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """

    def helper(in_left, in_right):
        global pre_index
        if in_left == in_right:
            return None
        root_val = preorder[pre_index]
        pre_index += 1
        root = TreeNode(root_val)
        index = val_map[root_val]
        root.left = helper(in_left, index)
        root.right = helper(index + 1, in_right)
        return root

    global pre_index
    pre_index = 0
    val_map = {kv[1]: kv[0] for kv in enumerate(inorder)}
    return helper(0, len(preorder))


preorder = [3, 9, 20, 15, 7]
inorder = [9, 3, 15, 20, 7]

tree = buildTree(preorder, inorder)
print tree
pretty_print(tree)