# -*- coding: utf-8 -*-
# @since : 2020-03-14 23:36 
# @author : wongleon
from common_data_structure import TreeNode, Tree, deque
import sys

reload(sys)
sys.setdefaultencoding('utf8')


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


def inorder(root, visited=None):
    if root is None:
        return
    if visited is None:
        visited = []
    inorder(root.left, visited)
    visited.append(root.val)
    inorder(root.right, visited)
    return visited


def preorder(root, visited=None):
    if root is None:
        return
    if visited is None:
        visited = []
    visited.append(root.val)
    preorder(root.left, visited)
    preorder(root.right, visited)
    return visited


def postorder(root, visited=None):
    if root is None:
        return
    if visited is None:
        visited = []
    postorder(root.left, visited)
    postorder(root.right, visited)
    visited.append(root.val)
    return visited


def level_traverse_bfs(root):
    visited = []
    queue = deque()
    queue.append(root)
    visited.append(root.val)

    while queue:
        p = queue.popleft()
        if p.left is not None:
            visited.append(p.left.val)
            queue.append(p.left)
        if p.right is not None:
            visited.append(p.right.val)
            queue.append(p.right)
    return visited


def max_depth(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if root is None:
        return 0
    ld = max_depth(root.left)
    rd = max_depth(root.right)
    return max(ld, rd) + 1


if __name__ == '__main__':
    tree = Tree()
    for i in xrange(5):
        tree.build_tree(i)

    print max_depth(tree.root)

    print "层次遍历(广度优先遍历):"
    print level_traverse_bfs(tree.root)
    print "前序遍历(深度优先遍历)："
    print preorder(tree.root)
    print "中序遍历："
    print inorder(tree.root)
    print "后序遍历："
    print postorder(tree.root)
