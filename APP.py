# import math
# import time
#
# import numpy as np
#
# import Constant
#
# Dict = {}
# coordinate = Dict.get(1) if Dict.__contains__(1) else []
# coordinate.append([1, 2])
# Dict.setdefault(1, coordinate)
# print(Dict)
#
# PRICE = [[1, 2]]
#
#
# def init(Price: []):
#     Price.append([1, 2])
#
#
# p = []
# init(p)
# print(p)
#
# a = 0b00000000
# i = 0
# print(a & (1 << i))
#
# for i in range(10):
#     print(i)
#     i += 1
#
#
#
# # if 3 and 0:
# #     print(1111111)
# #
# # class test:
# #     def __init__(self, dest: np.ndarray):
# #         self.t = dest
# #         self.v = 0
# #
# #     def modify(self, va: int):
# #         self.v = va
# #
# #
# # t = test(np.array([1, 2]))
# # print(t.__dict__)
# # print(np.array([1, 2]))
# #
# # w = test(np.array([1, 2]))
# # k = test(np.array([1, 2]))
# # s = [w]
# # if w in s:
# #     print(1)
# # elif k in s:
# #     print(2)
# #
# #
# # for i in range(1, 10):
# #     print(i)
# #
# # for i in range(2, 10):
# #     print(i)
# #
# # print(float('inf') > 1000)
#
# # import numpy as np
# #
# #
# # class KDTree:
# #     def __init__(self, data, objects):
# #         self.k = data.shape[1]  # 数据维度
# #         self.tree = self.build(data, objects)
# #
# #     class Node:
# #         def __init__(self, data, left, right, obj=None):
# #             self.data = data
# #             self.left = left
# #             self.right = right
# #             self.obj = obj
# #
# #     def build(self, data, objects, depth=0):
# #         if len(data) == 0:
# #             return None
# #         axis = depth % self.k
# #         sorted_idx = np.argsort(data[:, axis])
# #         data_sorted = data[sorted_idx]
# #         objects_sorted = objects[sorted_idx]
# #         mid = len(data) // 2
# #         return self.Node(data_sorted[mid],
# #                          self.build(data_sorted[:mid], objects_sorted[:mid], depth + 1),
# #                          self.build(data_sorted[mid + 1:], objects_sorted[mid + 1:], depth + 1),
# #                          objects_sorted[mid])
# #
# #     def query(self, x, k=1):
# #         def search_knn(node, x, k, heap):
# #             if node is None:
# #                 return
# #             dist = np.linalg.norm(node.data - x) + node.obj.v
# #             if len(heap) < k:
# #                 heap.append((dist, node.obj))
# #             elif dist < heap[-1][0]:
# #                 heap[-1] = (dist, node.obj)
# #             axis = node.data.argmax()  # 取最大值对应的维度
# #             if x[axis] < node.data[axis]:
# #                 search_knn(node.left, x, k, heap)
# #             else:
# #                 search_knn(node.right, x, k, heap)
# #
# #         heap = []
# #         search_knn(self.tree, x, k, heap)
# #         return sorted(heap)
# #
# #     def visualize(self, xmin, xmax, ymin, ymax):
# #         import matplotlib.pyplot as plt
# #         def plot_node(node, axis, xmin, xmax, ymin, ymax):
# #             if node is None:
# #                 return
# #             if axis == 0:
# #                 plt.plot([node.data[0], node.data[0]], [ymin, ymax], 'k--', linewidth=0.5)
# #                 plot_node(node.left, 1, xmin, node.data[0], ymin, ymax)
# #                 plot_node(node.right, 1, node.data[0], xmax, ymin, ymax)
# #             else:
# #                 plt.plot([xmin, xmax], [node.data[1], node.data[1]], 'k--', linewidth=0.5)
# #                 plot_node(node.left, 0, xmin, xmax, ymin, node.data[1])
# #                 plot_node(node.right, 0, xmin, xmax, node.data[1], ymax)
# #             plt.plot(node.data[0], node.data[1], 'ro', markersize=3)
# #
# #         plt.figure(figsize=(8, 8))
# #         plt.xlim(xmin, xmax)
# #         plt.ylim(ymin, ymax)
# #         plot_node(self.tree, 0, xmin, xmax, ymin, ymax)
# #         plt.show()
# #
# # # data = np.empty((0, 2))
# # # for i in range(100):
# # #     point = np.array([1, 2])
# # #     data = np.vstack([data, point])
# #
# # # 生成随机数据
# # data = np.empty((0, 2))
# # for i in range(100):
# #     data = np.vstack([data, np.random.rand(1, 2)])
# #
# # data1 = []
# # for i in range(100):
# #     point = np.array([random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)])
# #     data1.append(point)
# # data1 = np.array(data1)
# #
# # # print(data.shape)
# # # print(data1.shape)
# #
# # pointset = []
# # for i in range(100):
# #     t = test(np.array([1, 2]))
# #     pointset.append(t)
# # pointset = np.array(pointset)
# #
# # # 建树
# # tree = KDTree(data1, pointset)
# # # 查询
# # query_point = np.array([0.5, 0.5])
# # k_nearest = tree.query(query_point, k=3)
# # print(k_nearest)
# #
# # for i in pointset:
# #     i.v = 1
# #
# # # 查询
# # query_point = np.array([0.5, 0.5])
# # k_nearest = tree.query(query_point, k=3)
# # print(k_nearest[0][1])
# #
# # # 可视化
# # tree.visualize(0, 1, 0, 1)
# #
# # # a = np.array([1, 2])
# # # b = np.array([2, 3])
# # # print(b)
#
# # def change(s: []):
# #     s.append(test(np.array([1, 2])))
# #
# # yes = []
# # b = change(yes)
# # print(yes)
# #
# # print(Constant.INIT_SPEED)
# #
# # import math
# #
# #
# # def calculate_next_velocity(current_pos, current_speed, current_heading, target_pos):
# #     # 计算目标方向和距离
# #     target_direction = math.atan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0])
# #     target_distance = math.sqrt((target_pos[0] - current_pos[0]) ** 2 + (target_pos[1] - current_pos[1]) ** 2)
# #
# #     # 如果已经到达终点
# #     if target_distance < 0.4:
# #         return 0, 0
# #
# #     # 计算角速度
# #     heading_error = target_direction - current_heading
# #     if heading_error > math.pi:
# #         heading_error -= 2 * math.pi
# #     elif heading_error < -math.pi:
# #         heading_error += 2 * math.pi
# #     max_torque = 50
# #     angular_acceleration = heading_error * max_torque / 20  # 角加速度 = 角误差 * 最大力矩 / (密度 * 半径^2)
# #     max_angular_speed = math.pi
# #     if angular_acceleration > 0:
# #         max_angular_speed = min(max_angular_speed, math.sqrt(2 * angular_acceleration * math.pi / 50))
# #     else:
# #         max_angular_speed = max(-max_angular_speed, -math.sqrt(2 * abs(angular_acceleration) * math.pi / 50))
# #     angular_speed = max(min(max_angular_speed, math.pi), -math.pi)
# #
# #     # 计算线速度
# #     max_force = 250
# #     max_speed_forward = 6
# #     max_speed_backward = -2
# #     speed_error = min(max_speed_forward, max_speed_backward, target_distance) - current_speed
# #     linear_acceleration = speed_error * max_force / 20  # 线加速度 = 速度误差 * 最大牵引力 / 密度
# #     if linear_acceleration > 0:
# #         max_linear_speed = min(max_speed_forward, math.sqrt(2 * linear_acceleration * 6))
# #     else:
# #         max_linear_speed = max(max_speed_backward, -math.sqrt(2 * abs(linear_acceleration) * 2))
# #     linear_speed = max(min(max_linear_speed, 6), -2)
# #
# #     return angular_speed, linear_speed
# #
# # s = time.perf_counter()
# # calculate_next_velocity([3,3], 0, 0, [3,10])
# # e = time.perf_counter()
# # print(e - s)
# #
# # di = {"sss": -1}
# # for key, value in di.items():
# #     print(key == 'sss')
# #     print(value)
#
# # class test:
# #     def __init__(self, coordinate):
# #         self.coordinate = coordinate
# #
# #
# # class KDTree:
# #     def __init__(self, data, objects):
# #         self.k = data.shape[1]  # 数据维度
# #         self.tree = self.build(data, objects)
# #
# #     class Node:
# #         def __init__(self, data, left, right, obj=None):
# #             self.data = data
# #             self.left = left
# #             self.right = right
# #             self.obj = obj
# #
# #     def build(self, data, objects, depth=0):
# #         if len(data) == 0:
# #             return None
# #         axis = depth % self.k
# #         sorted_idx = np.argsort(data[:, axis])
# #         data_sorted = data[sorted_idx]
# #         objects_sorted = objects[sorted_idx]
# #         mid = len(data) // 2
# #         return self.Node(data_sorted[mid],
# #                          self.build(data_sorted[:mid], objects_sorted[:mid], depth + 1),
# #                          self.build(data_sorted[mid + 1:], objects_sorted[mid + 1:], depth + 1),
# #                          objects_sorted[mid])
# #
# #     def query(self, obj_x, k=1, filter_func=None):
# #         def search_knn(node, obj_x, k, heap):
# #             if node is None:
# #                 return
# #             filter_flag = filter_func(obj_x, node.obj)
# #             x = obj_x.coordinate
# #             dist = np.linalg.norm(node.data - x)
# #             if not filter_flag:
# #                 if len(heap) < k:
# #                     heap.append((dist, node.obj))
# #                 elif dist < heap[-1][0]:
# #                     heap[-1] = (dist, node.obj)
# #             axis = node.data.argmax()  # 取最大值对应的维度
# #             if x[axis] < node.data[axis]:
# #                 search_knn(node.left, obj_x, k, heap)
# #                 if len(heap) < k or abs(node.data[axis] - x[axis]) < heap[-1][0]:
# #                     search_knn(node.right, obj_x, k, heap)
# #             else:
# #                 search_knn(node.right, obj_x, k, heap)
# #                 if len(heap) < k or abs(node.data[axis] - x[axis]) < heap[-1][0]:
# #                     search_knn(node.left, obj_x, k, heap)
# #
# #         heap = []
# #         search_knn(self.tree, obj_x, k, heap)
# #         return sorted(heap, key=lambda res: res[0])
# #
# # def fil(n1, n2):
# #     if n1.coordinate[0] > 8:
# #         return True
# #     return False
# #
# # te = []
# # for i in range(10):
# #     te.append(test(np.array([1, 2])))
# # point = np.random.random((10, 2))
# # tree = KDTree(point, np.array(te))
# # print(tree.query(test(np.array([1, 2])), 8, fil))
#
# # coord_r = np.array([44.38, 2.53])
# # coord_d = np.array([8.75, 19.75])
# # target_angle = math.atan2(coord_d[1] - coord_r[1], coord_d[0] - coord_r[0])
# # heading_error = 0
# # if target_angle >= -0.49511 or target_angle <= -0.49511 - math.pi:
# #     heading_error = (target_angle + 0.49511 + 2 * math.pi) % math.pi
# # else:
# #     heading_error = -((target_angle + 0.49511 + 2 * math.pi) % math.pi)
# #
# # print(heading_error)
# #
# # angle1 = math.atan2(-1, 1)
# # print(angle1 == -math.pi/4)
# #
# # import numpy as np
# # from math import *
# # import matplotlib.pyplot as plt
# #
# # # 参数设置
# # V_Min = -0.5  # 最小速度
# # V_Max = 3.0  # 最大速度
# # W_Min = -50 * pi / 180.0  # 最小角速度
# # W_Max = 50 * pi / 180.0  # 最大角速度
# # Va = 0.5  # 加速度
# # Wa = 30.0 * pi / 180.0  # 角加速度
# # Vreso = 0.01  # 速度分辨率
# # Wreso = 0.1 * pi / 180.0  # 角速度分辨率
# # radius = 1  # 机器人模型半径
# # Dt = 0.1  # 时间间隔
# # Predict_Time = 4.0  # 模拟轨迹的持续时间
# # alpha = 1.0  # 距离目标点的评价函数的权重系数
# # Belta = 1.0  # 速度评价函数的权重系数
# # Gamma = 1.0  # 距离障碍物距离的评价函数的权重系数
# #
# # # 障碍物
# # Obstacle = np.array(
# #     [[0, 10], [2, 10], [4, 10], [6, 10], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5], [10, 7], [10, 9], [10, 11],
# #      [10, 13]])
# #
# #
# # # Obstacle = np.array([[0, 2]])
# #
# # # 距离目标点的评价函数
# # def Goal_Cost(Goal, Pos):
# #     return sqrt((Pos[-1, 0] - Goal[0]) ** 2 + (Pos[-1, 1] - Goal[1]) ** 2)
# #
# #
# # # 速度评价函数
# # def Velocity_Cost(Pos):
# #     return V_Max - Pos[-1, 3]
# #
# #
# # # 距离障碍物距离的评价函数
# # def Obstacle_Cost(Pos, Obstacle):
# #     MinDistance = float('Inf')  # 初始化时候机器人周围无障碍物所以最小距离设为无穷
# #     for i in range(len(Pos)):  # 对每一个位置点循环
# #         for j in range(len(Obstacle)):  # 对每一个障碍物循环
# #             Current_Distance = sqrt(
# #                 (Pos[i, 0] - Obstacle[j, 0]) ** 2 + (Pos[i, 1] - Obstacle[j, 1]) ** 2)  # 求出每个点和每个障碍物距离
# #             if Current_Distance < radius:  # 如果小于机器人自身的半径那肯定撞到障碍物了返回的评价值自然为无穷
# #                 return float('Inf')
# #             if Current_Distance < MinDistance:
# #                 MinDistance = Current_Distance  # 得到点和障碍物距离的最小
# #
# #     return 1 / MinDistance
# #
# #
# # # 速度采用
# # def V_Range(X):
# #     Vmin_Actual = X[3] - Va * Dt  # 实际在dt时间间隔内的最小速度
# #     Vmax_actual = X[3] + Va * Dt  # 实际载dt时间间隔内的最大速度
# #     Wmin_Actual = X[4] - Wa * Dt  # 实际在dt时间间隔内的最小角速度
# #     Wmax_Actual = X[4] + Wa * Dt  # 实际在dt时间间隔内的最大角速度
# #     VW = [max(V_Min, Vmin_Actual), min(V_Max, Vmax_actual), max(W_Min, Wmin_Actual),
# #           min(W_Max, Wmax_Actual)]  # 因为前面本身定义了机器人最小最大速度所以这里取交集
# #     return VW
# #
# #
# # # 一条模拟轨迹路线中的位置，速度计算
# # def Motion(X, u, dt):
# #     X[0] += u[0] * dt * cos(X[2])  # x方向上位置
# #     X[1] += u[0] * dt * sin(X[2])  # y方向上位置
# #     X[2] += u[1] * dt  # 角度变换
# #     X[3] = u[0]  # 速度
# #     X[4] = u[1]  # 角速度
# #     return X
# #
# #
# # # 一条模拟轨迹的完整计算
# # def Calculate_Traj(X, u):
# #     Traj = np.array(X)
# #     Xnew = np.array(X)
# #     time = 0
# #     while time <= Predict_Time:  # 一条模拟轨迹时间
# #         Xnew = Motion(Xnew, u, Dt)
# #         Traj = np.vstack((Traj, Xnew))  # 一条完整模拟轨迹中所有信息集合成一个矩阵
# #         time = time + Dt
# #     return Traj
# #
# #
# # # DWA核心计算
# # def dwa_Core(X, u, goal, obstacles):
# #     vw = V_Range(X)
# #     best_traj = np.array(X)
# #     min_score = 10000.0  # 随便设置一下初始的最小评价分数
# #     for v in np.arange(vw[0], vw[1], Vreso):  # 对每一个线速度循环
# #         for w in np.arange(vw[2], vw[3], Wreso):  # 对每一个角速度循环
# #             traj = Calculate_Traj(X, [v, w])
# #             goal_score = Goal_Cost(goal, traj)
# #             vel_score = Velocity_Cost(traj)
# #             obs_score = Obstacle_Cost(traj, Obstacle)
# #             score = goal_score + vel_score + obs_score
# #             if min_score >= score:  # 得出最优评分和轨迹
# #                 min_score = score
# #                 u = np.array([v, w])
# #                 best_traj = traj
# #
# #     return u, best_traj
# #
# #
# # x = np.array([2, 2, 45 * pi / 180, 0, 0])  # 设定初始位置，角速度，线速度
# # u = np.array([0, 0])  # 设定初始速度
# # goal = np.array([8, 8])  # 设定目标位置
# # global_tarj = np.array(x)
# # for i in range(1000):  # 循环1000次，这里可以直接改成while的直到循环到目标位置
# #     u, current = dwa_Core(x, u, goal, Obstacle)
# #     x = Motion(x, u, Dt)
# #     global_tarj = np.vstack((global_tarj, x))  # 存储最优轨迹为一个矩阵形式每一行存放每一条最有轨迹的信息
# #     if sqrt((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2) <= radius:  # 判断是否到达目标点
# #         print('Arrived')
# #         break
# #
# # plt.plot(global_tarj[:, 0], global_tarj[:, 1], '*r', Obstacle[0:3, 0], Obstacle[0:3, 1], '-g', Obstacle[4:9, 0],
# #          Obstacle[4:9, 1], '-g', Obstacle[10:13, 0], Obstacle[10:13, 1], '-g')  # 画出最优轨迹的路线
# # plt.show()
#
# import copy
#
# import numpy as np
# import matplotlib.pyplot as plt
# import math
#
#
# class Info():
#     def __init__(self):
#         # define robot move speed ,accelerate,radius ...and so on
#         # 定义机器人移动极限速度、加速度等信息
#         self.v_min = -0.5
#         self.v_max = 3.0
#         self.w_max = 50.0 * math.pi / 180.0
#         self.w_min = -50.0 * math.pi / 180.0
#         self.vacc_max = 0.5
#         self.wacc_max = 30.0 * math.pi / 180.0
#         self.v_reso = 0.01
#         self.w_reso = 0.1 * math.pi / 180.0
#         self.radius = 1.0
#         self.dt = 0.8#wxw
#         # self.dt = 0.1
#         self.predict_time = 4.0
#         self.goal_factor = 1.0
#         self.vel_factor = 1.0
#         self.traj_factor = 1.0
#
#
# # 定义机器人运动模型
# # 返回坐标(x,y),偏移角theta,速度v,角速度w
# def motion_model(x, u, dt):
#     # robot motion model: x,y,theta,v,w
#     x[0] += u[0] * dt * math.cos(x[2])
#     x[1] += u[0] * dt * math.sin(x[2])
#     x[2] += u[1] * dt
#     x[3] = u[0]
#     x[4] = u[1]
#     return x
#
#
# # 产生速度空间
# def vw_generate(x, info):
#     # generate v,w window for traj prediction
#     Vinfo = [info.v_min, info.v_max,
#              info.w_min, info.w_max]
#
#     Vmove = [x[3] - info.vacc_max * info.dt,
#              x[3] + info.vacc_max * info.dt,
#              x[4] - info.wacc_max * info.dt,
#              x[4] + info.wacc_max * info.dt]
#
#     # 保证速度变化不超过info限制的范围
#     vw = [max(Vinfo[0], Vmove[0]), min(Vinfo[1], Vmove[1]),
#           max(Vinfo[2], Vmove[2]), min(Vinfo[3], Vmove[3])]
#
#     return vw
#
#
# # 依据当前位置及速度，预测轨迹
# def traj_cauculate(x, u, info):
#     ctraj = np.array(x)
#     xnew = np.array(x)  # Caution!!! Don't use like this: xnew = x, it will change x value when run motion_modle below
#     time = 0
#
#     while time <= info.predict_time:  # preditc_time作用？循环40次
#         xnew = motion_model(xnew, u, info.dt)
#         ctraj = np.vstack((ctraj, xnew))
#         time += info.dt#0.1
#
#     return ctraj
#
#
# def dwa_core(x, u, goal, info, obstacles):
#     # the kernel of dwa
#     vw = vw_generate(x, info)
#     best_ctraj = np.array(x)
#     min_score = 10000.0
#
#     trajs = []
#
#     # 速度v,w都被限制在速度空间里
#     for v in np.arange(vw[0], vw[1], info.v_reso):
#         for w in np.arange(vw[2], vw[3], info.w_reso):
#             # cauculate traj for each given (v,w)
#             ctraj = traj_cauculate(x, [v, w], info)
#             # 计算评价函数
#             goal_score = info.goal_factor * goal_evaluate(ctraj, goal)
#             vel_score = info.vel_factor * velocity_evaluate(ctraj, info)
#             traj_score = info.traj_factor * traj_evaluate(ctraj, obstacles, info)
#             # 可行路径不止一条，通过评价函数确定最佳路径
#             # 路径总分数 = 距离目标点 + 速度 + 障碍物
#             # 分数越低，路径越优
#             ctraj_score = goal_score + vel_score + traj_score
#             # evaluate current traj (the score smaller,the traj better)
#             if min_score >= ctraj_score:
#                 min_score = ctraj_score
#                 u = np.array([v, w])
#                 best_ctraj = ctraj
#                 trajs.append(ctraj)
#
#     plt.ion()
#     plt.plot(goal[0], goal[1], 'or', markersize=5)
#     plt.plot([0, 14], [0, 0], '-k', linewidth=7)
#     plt.plot([0, 14], [14, 14], '-k', linewidth=7)
#     plt.plot([0, 0], [0, 14], '-k', linewidth=7)
#     plt.plot([14, 14], [0, 14], '-k', linewidth=7)
#     plt.plot([0, 6], [10, 10], '-y', linewidth=10)
#     plt.plot([3, 8], [5, 5], '-y', linewidth=10)
#     plt.plot([10, 10], [7, 13], '-y', linewidth=10)
#     plt.plot(obstacles[:, 0], obstacles[:, 1], '*b', markersize=10)
#     plt.plot(x[0], x[1], 'ob', markersize=5)
#     plt.arrow(x[0], x[1], math.cos(x[2]), math.sin(x[2]), width=0.02, fc='red')
#     plt.grid(True)
#
#     for id, t in enumerate(trajs):
#         if id %30 == 0:
#             plt.plot(t[:, 0], t[:, 1], '-r', linewidth=1)
#
#     plt.ioff()
#     plt.show()
#
#     return u, best_ctraj
#
#
# # 距离目标点评价函数
# def goal_evaluate(traj, goal):
#     # cauculate current pose to goal with euclidean distance
#     goal_score = math.sqrt((traj[-1, 0] - goal[0]) ** 2 + (traj[-1, 1] - goal[1]) ** 2)
#     return goal_score
#
#
# # 速度评价函数
# def velocity_evaluate(traj, info):
#     # cauculate current velocty score
#     vel_score = info.v_max - traj[-1, 3]
#     return vel_score
#
#
# # 轨迹距离障碍物的评价函数
# def traj_evaluate(traj, obstacles, info):
#     # evaluate current traj with the min distance to obstacles
#     min_dis = float("Inf")
#     for i in range(len(traj)):
#         for ii in range(len(obstacles)):
#             current_dist = math.sqrt((traj[i, 0] - obstacles[ii, 0]) ** 2 + (traj[i, 1] - obstacles[ii, 1]) ** 2)
#
#             if current_dist <= info.radius:
#                 return float("Inf")
#
#             if min_dis >= current_dist:
#                 min_dis = current_dist
#
#     return 1 / min_dis
#
#
# # 生成包含障碍物的地图
# def obstacles_generate():
#     #	Map shape and obstacles:
#     #	 ___________________________________
#     #	|                                   |
#     #	|                                   |
#     #	|____________             |         |
#     #	|                         |         |
#     #	|                   goal  |         |
#     #	|                   O     |         |
#     #	|                         |         |
#     #	|                                   |
#     #	|       _____________               |
#     #	|                                   |
#     #	|     *(start)                      |
#     #	|                                   |
#     #	|___________________________________|
#
#     obstacles = np.array([[0, 10],
#                           [2, 10],
#                           [4, 10],
#                           [6, 10],
#                           [3, 5],
#                           [4, 5],
#                           [5, 5],
#                           [6, 5],
#                           [7, 5],
#                           [8, 5],
#                           [10, 7],
#                           [10, 9],
#                           [10, 11],
#                           [10, 13]])
#     return obstacles
#
# def local_traj_display(x, goal, current_traj, obstacles):
#     # display current pose ,traj prodicted,map,goal
#     plt.cla()
#     plt.plot(goal[0], goal[1], 'or', markersize=1)
#     plt.plot([0, 14], [0, 0], '-k', linewidth=7)
#     plt.plot([0, 14], [14, 14], '-k', linewidth=7)
#     plt.plot([0, 0], [0, 14], '-k', linewidth=7)
#     plt.plot([14, 14], [0, 14], '-k', linewidth=7)
#     plt.plot([0, 6], [10, 10], '-y', linewidth=10)
#     plt.plot([3, 8], [5, 5], '-y', linewidth=10)
#     plt.plot([10, 10], [7, 13], '-y', linewidth=10)
#     plt.plot(obstacles[:, 0], obstacles[:, 1], '*b', markersize=8)
#     plt.plot(x[0], x[1], 'ob', markersize=10)
#     plt.arrow(x[0], x[1], math.cos(x[2]), math.sin(x[2]), width=0.02, fc='red')
#     plt.plot(current_traj[:, 0], current_traj[:, 1], '-g', linewidth=2)
#     plt.grid(True)
#     plt.pause(0.001)
#
# def main():
#     x = np.array([2, 2, 45 * math.pi / 180, 0, 0])
#     u = np.array([0, 0])
#     goal = np.array([12, 4])
#     info = Info()
#     obstacles = obstacles_generate()
#     global_traj = np.array(x)
#     # plt.figure('DWA Algorithm')
#
#     while 1:
#         u, current_traj = dwa_core(x, u, goal, info, obstacles)
#         x = motion_model(x, u, info.dt)
#         global_traj = np.vstack((global_traj, x))
#         # local_traj_display(x, goal, current_traj, obstacles)
#         if math.sqrt((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2) <= info.radius:
#             print("Goal Arrived!")
#             break
#
#     # plt.plot(global_traj[:, 0], global_traj[:, 1], '-r')
#     # plt.show()
#
#
# if __name__ == "__main__":
#     main()
import math

import numpy as np


def maybe_collide(r1, r2) -> bool:
    if math.fabs(r1.aspect - r2.aspect) == math.pi:
        return False
    elif r1.aspect == r2.aspect:
        return False
    else:
        x0, y0 = -1, -1
        if r1.aspect == math.pi / 2 and r2.aspect != math.pi / 2:
            x0 = r1.coordinate[0]
            point2 = r2.coordinate
            asp2 = r2.aspect if r2.aspect >= 0 else math.pi + r2.aspect
            k2 = math.tan(asp2)
            b2 = point2[1] - k2 * point2[0]
            y0 = k2 * x0 + b2
        elif r2.aspect == math.pi / 2 and r1.aspect != math.pi / 2:
            x0 = r2.coordinate[0]
            point1 = r1.coordinate
            asp1 = r1.aspect if r1.aspect >= 0 else math.pi + r1.aspect
            k1 = math.tan(asp1)
            b1 = point1[1] - k1 * point1[0]
            y0 = k1 * x0 + b1
        elif r1.aspect == math.pi / 2 and r2.aspect == math.pi / 2:
            return False
        else:
            asp1 = r1.aspect if r1.aspect >= 0 else math.pi + r1.aspect
            asp2 = r2.aspect if r2.aspect >= 0 else math.pi + r2.aspect
            k1 = math.tan(asp1)
            k2 = math.tan(asp2)
            point1 = r1.coordinate
            point2 = r2.coordinate
            b1 = point1[1] - k1 * point1[0]
            b2 = point2[1] - k2 * point2[0]

            x0 = (b2 - b1) / (k1 - k2)
            y0 = k1 * x0 + b1
            print(x0)
            print(y0)

        if x0 > 0 and y0 > 0:
            angle1 = math.atan2(y0 - r1.coordinate[1], x0 - r1.coordinate[0])
            angle2 = math.atan2(y0 - r2.coordinate[1], x0 - r2.coordinate[0])
            point_side1 = True if angle1 == r1.aspect else False
            point_side2 = True if angle2 == r2.aspect else False
            return point_side1 and point_side2
        else:
            return False


class test:
    def __init__(self, coordinate, aspect):
        self.coordinate = coordinate
        self.aspect = aspect


r1 = test(np.array([13.70, 14.29]), 0.713296)
r2 = test(np.array([15.15, 14.45]), 1.617699)
print(maybe_collide(r1, r2))

# if math.fabs(r1.aspect - r2.aspect) == math.pi or r1.aspect == r2.aspect:
#     if np.linalg.norm(r1.coordinate - r2.coordinate) <= r1.radius + r2.radius + const.COLLIDE_DISTANCE:
#         return True
#     else:
#         return False
# else:
#     x0, y0 = -1, -1
#     if r1.aspect == math.pi / 2 and r2.aspect != math.pi / 2:
#         x0 = r1.coordinate[0]
#         point2 = r2.coordinate
#         asp2 = r2.aspect if r2.aspect >= 0 else math.pi + r2.aspect
#         k2 = math.tan(asp2)
#         b2 = point2[1] - k2 * point2[0]
#         y0 = k2 * x0 + b2
#     elif r2.aspect == math.pi / 2 and r1.aspect != math.pi / 2:
#         x0 = r2.coordinate[0]
#         point1 = r1.coordinate
#         asp1 = r1.aspect if r1.aspect >= 0 else math.pi + r1.aspect
#         k1 = math.tan(asp1)
#         b1 = point1[1] - k1 * point1[0]
#         y0 = k1 * x0 + b1
#     elif r1.aspect == math.pi / 2 and r2.aspect == math.pi / 2:
#         return False
#     else:
#         asp1 = r1.aspect if r1.aspect >= 0 else math.pi + r1.aspect
#         asp2 = r2.aspect if r2.aspect >= 0 else math.pi + r2.aspect
#         k1 = math.tan(asp1)
#         k2 = math.tan(asp2)
#         point1 = r1.coordinate
#         point2 = r2.coordinate
#         b1 = point1[1] - k1 * point1[0]
#         b2 = point2[1] - k2 * point2[0]
#
#         x0 = (b2 - b1) / (k1 - k2)
#         y0 = k1 * x0 + b1
#
#     if x0 > 0 and y0 > 0:
#         angle1 = math.atan2(y0 - r1.coordinate[1], x0 - r1.coordinate[0])
#         angle2 = math.atan2(y0 - r2.coordinate[1], x0 - r2.coordinate[0])
#         point_side1 = True if angle1 == r1.aspect else False
#         point_side2 = True if angle2 == r2.aspect else False
#         return point_side1 and point_side2
#     else:
#         return False