import time

import numpy as np

import Constant

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

a = 0b00000000
i = 0
print(a & (1 << i))

for i in range(10):
    print(i)
    i += 1



# if 3 and 0:
#     print(1111111)
#
# class test:
#     def __init__(self, dest: np.ndarray):
#         self.t = dest
#         self.v = 0
#
#     def modify(self, va: int):
#         self.v = va
#
#
# t = test(np.array([1, 2]))
# print(t.__dict__)
# print(np.array([1, 2]))
#
# w = test(np.array([1, 2]))
# k = test(np.array([1, 2]))
# s = [w]
# if w in s:
#     print(1)
# elif k in s:
#     print(2)
#
#
# for i in range(1, 10):
#     print(i)
#
# for i in range(2, 10):
#     print(i)
#
# print(float('inf') > 1000)

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
# print(k_nearest[0][1])
#
# # 可视化
# tree.visualize(0, 1, 0, 1)
#
# # a = np.array([1, 2])
# # b = np.array([2, 3])
# # print(b)

# def change(s: []):
#     s.append(test(np.array([1, 2])))
#
# yes = []
# b = change(yes)
# print(yes)
#
# print(Constant.INIT_SPEED)
#
# import math
#
#
# def calculate_next_velocity(current_pos, current_speed, current_heading, target_pos):
#     # 计算目标方向和距离
#     target_direction = math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
#     target_distance = math.sqrt((target_pos[0] - current_pos[0]) ** 2 + (target_pos[1] - current_pos[1]) ** 2)
#
#     # 如果已经到达终点
#     if target_distance < 0.4:
#         return 0, 0
#
#     # 计算角速度
#     heading_error = target_direction - current_heading
#     if heading_error > math.pi:
#         heading_error -= 2 * math.pi
#     elif heading_error < -math.pi:
#         heading_error += 2 * math.pi
#     max_torque = 50
#     angular_acceleration = heading_error * max_torque / 20  # 角加速度 = 角误差 * 最大力矩 / (密度 * 半径^2)
#     max_angular_speed = math.pi
#     if angular_acceleration > 0:
#         max_angular_speed = min(max_angular_speed, math.sqrt(2 * angular_acceleration * math.pi / 50))
#     else:
#         max_angular_speed = max(-max_angular_speed, -math.sqrt(2 * abs(angular_acceleration) * math.pi / 50))
#     angular_speed = max(min(max_angular_speed, math.pi), -math.pi)
#
#     # 计算线速度
#     max_force = 250
#     max_speed_forward = 6
#     max_speed_backward = -2
#     speed_error = min(max_speed_forward, max_speed_backward, target_distance) - current_speed
#     linear_acceleration = speed_error * max_force / 20  # 线加速度 = 速度误差 * 最大牵引力 / 密度
#     if linear_acceleration > 0:
#         max_linear_speed = min(max_speed_forward, math.sqrt(2 * linear_acceleration * 6))
#     else:
#         max_linear_speed = max(max_speed_backward, -math.sqrt(2 * abs(linear_acceleration) * 2))
#     linear_speed = max(min(max_linear_speed, 6), -2)
#
#     return angular_speed, linear_speed
#
# s = time.perf_counter()
# calculate_next_velocity([3,3], 0, 0, [3,10])
# e = time.perf_counter()
# print(e - s)
#
# di = {"sss": -1}
# for key, value in di.items():
#     print(key == 'sss')
#     print(value)

# class test:
#     def __init__(self, coordinate):
#         self.coordinate = coordinate
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
#     def query(self, obj_x, k=1, filter_func=None):
#         def search_knn(node, obj_x, k, heap):
#             if node is None:
#                 return
#             filter_flag = filter_func(obj_x, node.obj)
#             x = obj_x.coordinate
#             dist = np.linalg.norm(node.data - x)
#             if not filter_flag:
#                 if len(heap) < k:
#                     heap.append((dist, node.obj))
#                 elif dist < heap[-1][0]:
#                     heap[-1] = (dist, node.obj)
#             axis = node.data.argmax()  # 取最大值对应的维度
#             if x[axis] < node.data[axis]:
#                 search_knn(node.left, obj_x, k, heap)
#                 if len(heap) < k or abs(node.data[axis] - x[axis]) < heap[-1][0]:
#                     search_knn(node.right, obj_x, k, heap)
#             else:
#                 search_knn(node.right, obj_x, k, heap)
#                 if len(heap) < k or abs(node.data[axis] - x[axis]) < heap[-1][0]:
#                     search_knn(node.left, obj_x, k, heap)
#
#         heap = []
#         search_knn(self.tree, obj_x, k, heap)
#         return sorted(heap, key=lambda res: res[0])
#
# def fil(n1, n2):
#     if n1.coordinate[0] > 8:
#         return True
#     return False
#
# te = []
# for i in range(10):
#     te.append(test(np.array([1, 2])))
# point = np.random.random((10, 2))
# tree = KDTree(point, np.array(te))
# print(tree.query(test(np.array([1, 2])), 8, fil))

k = [1, 2, 3]
for i in k:
    print(i)
print(k[0])