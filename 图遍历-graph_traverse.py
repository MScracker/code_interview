# -*- coding: utf-8 -*- 
# @since : 2020-03-14 15:12 
# @author : wongleon

from common_data_structure import *
from collections import deque


def depth_first_search(g, root):
    if not isinstance(g, Graph) or root is None or root not in g.nodes():
        raise Exception('input params error!')
    visited = []  # 存放访问顺序同时用作不重复访问判断
    stack = deque()  # 用作堆栈
    stack.append(root)
    visited.append(root)

    while stack:
        current_node = stack.pop()
        for neighbour in g.node_neighbors[current_node]:
            if neighbour not in visited:
                stack.append(current_node)
                stack.append(neighbour)
                visited.append(neighbour)  # 保存访问顺序
                break  # 为保持深度访问优先

    return visited


def recursive_depth_first_search(g, root, visited=None):
    if not isinstance(g, Graph) or root not in g.nodes():
        raise Exception('input params error!')
    if visited is None:
        visited = []
    visited.append(root)
    for neighbour in g.node_neighbors[root]:
        if neighbour not in visited:  # 是否访问过
            recursive_depth_first_search(g, neighbour, visited)
    return visited


def breadth_first_search(g, root):
    if not isinstance(g, Graph) or root not in g.nodes():
        raise Exception('input params error!')
    visited = []  # 存放访问顺序
    queue = deque()  # 用作队列
    visited.append(root)
    queue.append(root)

    while queue:
        current_node = queue.popleft()
        for neighbour in g.node_neighbors[current_node]:
            if neighbour not in visited:
                visited.append(neighbour)  # 保存访问顺序
                queue.append(neighbour)
    return visited


g = Graph()
g.add_link((1, [2, 3]))
g.add_link((2, [4, 5, 1]))
g.add_link((4, [8, 2]))
g.add_link((8, [5, 4]))
g.add_link((5, [2, 8]))
g.add_link((3, [6, 7, 1]))
g.add_link((6, [7, 3]))
g.add_link((7, [3, 6]))

print '迭代DFS遍历：'
print depth_first_search(g, 1)
print '递归DFS遍历：'
print recursive_depth_first_search(g, 1)
print '迭代BFS遍历:'
print breadth_first_search(g, 1)
