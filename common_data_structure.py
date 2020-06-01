# -- coding: UTF-8 --
# @since : 2020-03-14 14:31
# @author : wongleon
import sys

reload(sys)
sys.setdefaultencoding('utf8')
from collections import deque
from io import StringIO
import math


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


def build_tree_from_list(values):
    def recursive(root, values, depth):
        if not values:
            raise Exception('value list is is_empty')
        if depth < len(values):
            if values[depth] is None:
                return None
            else:
                if root is None:
                    root = TreeNode(values[depth])
                root.left = recursive(root.left, values, 2 * depth + 1)
                root.right = recursive(root.right, values, 2 * depth + 2)
            return root

    return recursive(None, values, 0)


class Queue(object):
    def __init__(self):
        self.queue = []

    def enqueue(self, b):
        self.queue.insert(0, b)

    def dequeue(self):
        return self.queue.pop()

    def isEmpty(self):
        return self.queue == []


def add_padding(str, pad_length_value):
    str = str.strip()
    return str.center(pad_length_value, ' ')


def pretty_print(tree):
    output = StringIO()
    pretty_output = StringIO()

    current_level = Queue()
    next_level = Queue()
    current_level.enqueue(tree)
    depth = 0

    # get the depth of current tree
    # get the tree node data and store in list
    if tree:
        while not current_level.isEmpty():
            current_node = current_level.dequeue()
            output.write('%s ' % unicode(current_node.val) if current_node else u'N ')
            next_level.enqueue(
                current_node.left if current_node else current_node)
            next_level.enqueue(
                current_node.right if current_node else current_node)

            if current_level.isEmpty():
                if sum([i is not None for i in next_level.queue]
                       ):  # if next level has node
                    current_level, next_level = next_level, current_level
                    depth = depth + 1
                output.write(u'\n')
    print('the tree print level by level is :')
    print(output.getvalue())
    print("current tree's depth is %i" % (depth + 1))

    # add space to each node
    output.seek(0)
    pad_length = 3
    keys = []
    spaces = int(math.pow(2, depth))

    while spaces > 0:
        skip_start = spaces * pad_length
        skip_mid = (2 * spaces - 1) * pad_length

        key_start_spacing = u' ' * skip_start
        key_mid_spacing = u' ' * skip_mid

        keys = output.readline().split(u' ')  # read one level to parse
        padded_keys = (add_padding(key, pad_length) for key in keys)
        padded_str = key_mid_spacing.join(padded_keys)
        complete_str = u''.join([key_start_spacing, padded_str])

        pretty_output.write(complete_str)

        # add space and slashes to middle layer
        slashes_depth = spaces
        print('current slashes depth im_resize:')
        print(spaces)
        print("current levle's list is:")
        print(keys)
        spaces = spaces // 2
        if spaces > 0:
            pretty_output.write(u'\n')  # print '\n' each level

            cnt = 0
            while cnt < slashes_depth:
                inter_symbol_spacing = u' ' * (pad_length + 2 * cnt)
                symbol = ''.join([u'/', inter_symbol_spacing, u'\\'])
                symbol_start_spacing = u' ' * (skip_start - cnt - 1)
                symbol_mid_spacing = u' ' * (skip_mid - 2 * (cnt + 1))
                pretty_output.write(u''.join([symbol_start_spacing, symbol]))
                for i in keys[1:-1]:
                    pretty_output.write(u''.join([symbol_mid_spacing, symbol]))
                pretty_output.write(u'\n')
                cnt = cnt + 1

    print(pretty_output.getvalue())


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

    def __repr__(self):
        return 'TreeNode(..., val:' + str(self.val) + ', ...)'


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
