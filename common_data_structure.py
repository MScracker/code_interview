# -- coding: UTF-8 --
# @since : 2020-03-14 14:31
# @author : wongleon
import sys

reload(sys)
sys.setdefaultencoding('utf8')
from collections import deque


def obj_to_string(obj, cls):
    """
    简单地实现类似对象打印的方法
    :param obj: 对应类的实例
    :param cls: 对应的类(如果是继承的类也没有关系，比如A(object), cls参数传object一样适用，如果你不想这样，可以修改第一个if)
    :return: 实例对象的to_string
    """
    if not isinstance(obj, cls):
        raise TypeError("obj_to_string func: 'the object is not an instance of the specify class.'")
    to_string = str(cls.__name__) + "("
    items = obj.__dict__
    n = 0
    for k in items:
        if k.startswith("_"):  # or isinstance(items[k], cls):
            continue
        to_string = to_string + str(k) + "=" + str(items[k]) + ","
        n += 1
    if n == 0:
        to_string += str(cls.__name__).lower() + ": 'Instantiated objects have no property values'"
    return to_string.rstrip(",") + ")"


class ListNode:

    def __init__(self, data):
        self.val = data
        self.next = None

    def __repr__(self):
        return obj_to_string(self, ListNode)


class Graph:

    def __init__(self):
        self.node_neighbors = {}

    def add_link(self, kv):
        k, v = kv
        if isinstance(v, list):
            if k not in self.node_neighbors.keys():
                self.node_neighbors[k] = v
            else:
                self.node_neighbors[k] += v

    def add_links(self, kvs):
        for kv in kvs:
            self.add_link(kv)

    def nodes(self):
        return self.node_neighbors.keys()


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __str__(self):
        return obj_to_string(self, TreeNode)


class Tree:
    def __init__(self):
        self.root = None
        self.queue = deque()  # 保存树节点移动指针

    def build_tree(self, val):
        if self.root is None:
            self.root = TreeNode(val)
            self.queue.append(self.root)
        else:
            currentNode = TreeNode(val)
            pointer = self.queue[0]  # 找回左右节点没有结全
            if pointer.left is None:
                pointer.left = currentNode
                self.queue.append(currentNode)  # 将左右节点没长齐的新节点推入队列
            else:
                pointer.right = currentNode
                self.queue.append(currentNode)  # 将左右节点没长齐的新节点推入队列
                self.queue.popleft()  # 此时左右节点已满,退出队列


if __name__ == '__main__':
    tree = Tree()
    for i in xrange(5):
        print i
        tree.build_tree(i)

    node = TreeNode(1)
    node.left = TreeNode(2)
    node.right = TreeNode(3)
    print node
