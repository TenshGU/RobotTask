import random

import numpy as np

Dict = {}
coordinate = Dict.get(1) if Dict.__contains__(1) else []
coordinate.append([1, 2])
Dict.setdefault(1, coordinate)
print(Dict)

PRICE = [[1, 2]]


def init(Price: []):
    Price.append([1, 2])


p = []
init(p)
print(p)

a = 0b00000110
i = 3
print(a & (1 << i))

if 3 and 0:
    print(1111111)

class test:
    def __init__(self, dest: np.ndarray):
        self.t = dest
        self.v = 0

    def modify(self, va: int):
        self.v = va


t = test(np.array([1, 2]))
print(t.__dict__)
print(np.array([1, 2]))

w = test(np.array([1, 2]))
k = test(np.array([1, 2]))
s = [w]
if w in s:
    print(1)
elif k in s:
    print(2)


for i in range(1, 10):
    print(i)

for i in range(2, 10):
    print(i)

print(float('inf') > 1000)

# import numpy as np
#
#
# class KDTree:
#     def __init__(self, data, objects):
#         self.k = data.shape[1]  # 数据维度
#         self.tree = self.build(data, objects)
#
#     class Node:
#         def __init__(self, data, left, right, obj=None):
#             self.data = data
#             self.left = left
#             self.right = right
#             self.obj = obj
#
#     def build(self, data, objects, depth=0):
#         if len(data) == 0:
#             return None
#         axis = depth % self.k
#         sorted_idx = np.argsort(data[:, axis])
#         data_sorted = data[sorted_idx]
#         objects_sorted = objects[sorted_idx]
#         mid = len(data) // 2
#         return self.Node(data_sorted[mid],
#                          self.build(data_sorted[:mid], objects_sorted[:mid], depth + 1),
#                          self.build(data_sorted[mid + 1:], objects_sorted[mid + 1:], depth + 1),
#                          objects_sorted[mid])
#
#     def query(self, x, k=1):
#         def search_knn(node, x, k, heap):
#             if node is None:
#                 return
#             dist = np.linalg.norm(node.data - x) + node.obj.v
#             if len(heap) < k:
#                 heap.append((dist, node.obj))
#             elif dist < heap[-1][0]:
#                 heap[-1] = (dist, node.obj)
#             axis = node.data.argmax()  # 取最大值对应的维度
#             if x[axis] < node.data[axis]:
#                 search_knn(node.left, x, k, heap)
#             else:
#                 search_knn(node.right, x, k, heap)
#
#         heap = []
#         search_knn(self.tree, x, k, heap)
#         return sorted(heap)
#
#     def visualize(self, xmin, xmax, ymin, ymax):
#         import matplotlib.pyplot as plt
#         def plot_node(node, axis, xmin, xmax, ymin, ymax):
#             if node is None:
#                 return
#             if axis == 0:
#                 plt.plot([node.data[0], node.data[0]], [ymin, ymax], 'k--', linewidth=0.5)
#                 plot_node(node.left, 1, xmin, node.data[0], ymin, ymax)
#                 plot_node(node.right, 1, node.data[0], xmax, ymin, ymax)
#             else:
#                 plt.plot([xmin, xmax], [node.data[1], node.data[1]], 'k--', linewidth=0.5)
#                 plot_node(node.left, 0, xmin, xmax, ymin, node.data[1])
#                 plot_node(node.right, 0, xmin, xmax, node.data[1], ymax)
#             plt.plot(node.data[0], node.data[1], 'ro', markersize=3)
#
#         plt.figure(figsize=(8, 8))
#         plt.xlim(xmin, xmax)
#         plt.ylim(ymin, ymax)
#         plot_node(self.tree, 0, xmin, xmax, ymin, ymax)
#         plt.show()
#
# # data = np.empty((0, 2))
# # for i in range(100):
# #     point = np.array([1, 2])
# #     data = np.vstack([data, point])
#
# # 生成随机数据
# data = np.empty((0, 2))
# for i in range(100):
#     data = np.vstack([data, np.random.rand(1, 2)])
#
# data1 = []
# for i in range(100):
#     point = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
#     data1.append(point)
# data1 = np.array(data1)
#
# # print(data.shape)
# # print(data1.shape)
#
# pointset = []
# for i in range(100):
#     t = test(np.array([1, 2]))
#     pointset.append(t)
# pointset = np.array(pointset)
#
# # 建树
# tree = KDTree(data1, pointset)
# # 查询
# query_point = np.array([0.5, 0.5])
# k_nearest = tree.query(query_point, k=3)
# print(k_nearest)
#
# for i in pointset:
#     i.v = 1
#
# # 查询
# query_point = np.array([0.5, 0.5])
# k_nearest = tree.query(query_point, k=3)
# print(k_nearest)
#
# # 可视化
# tree.visualize(0, 1, 0, 1)
#
# # a = np.array([1, 2])
# # b = np.array([2, 3])
# # print(b)
