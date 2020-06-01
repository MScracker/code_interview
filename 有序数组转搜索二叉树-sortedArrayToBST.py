# -*- coding: utf-8 -*- 
# @since : 2020-03-15 03:08 
# @author : wongleon
import sys

reload(sys)
sys.setdefaultencoding('utf8')
from tree_traverse import TreeNode, inorder


def sortedArrayToBST(nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    n = len(nums)

    def recursive(l, r):
        if l > r:
            return None
        mid = (l + r) // 2
        root = TreeNode(nums[mid])
        root.left = recursive(l, mid - 1)
        root.right = recursive(mid + 1, r)
        return root

    return recursive(0, n - 1)


nums = [-10, -3, 0, 5, 9]

tree = sortedArrayToBST(nums)
print inorder(tree)
