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
#             dist = np.linalg.norm(node.data - x) + node.obj.future_value
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


# # construct the kd-tree
#     pointset = np.empty((0, 2))
#     workbenches_np = np.array(workbenches)
#     for workbench in workbenches:
#         point = workbench.coordinate
#         pointset = np.vstack([pointset, point])
#     tree = KDTree(pointset, workbenches_np)
#     kd_tree.append(tree)


# for index in sorted_indexes:
#     no_choose = True
#     # if robot has select a wb, or current robot has not grabbed by other robot
#     if robot in selected_r:
#         continue
#     k_nearest = k_nearest_dict[robot]
#     # choosing the best wb
#     for k in k_nearest:
#         distance, wb = k[0], k[1]
#         # if this wb has not grabbed by other robot, choose it
#         if wb not in selected_w:
#             selected[wb] = [robot, distance]
#             selected_w.add(wb)
#             selected_r.add(robot)
#             no_choose = False
#             break
#         # if this wb has grabbed by other robot, judge whether this robot distance less than pre-choose
#         # distance if more than it, continue to choose the next wb if less than, grab it, the pre-choose
#         # one will be removed from the selected_r after that, begin to the next robot
#         else:
#             old_robot = selected[wb][0]
#             if old_robot not in lock:
#                 if distance >= selected[wb][1]:
#                     continue
#                 else:
#                     selected[wb] = [robot, distance]
#                     selected_r.remove(old_robot)
#                     selected_r.add(robot)
#                     no_choose = False
#                     break
#
#     # force to choose one, and lock it
#     if no_choose:
#         distance, wb = k_nearest[0], k_nearest[1]
#         old_robot = selected[wb][0]
#         selected[wb] = [robot, distance]
#         selected_r.remove(old_robot)
#         selected_r.add(robot)
#         lock.append(robot)

# opt_dict = {}
#         # target_angle = robot.dest_wb.angle
#         # l_speed = math.sqrt(robot.line_speed_x ** 2 + robot.line_speed_y ** 2)
#         # a_speed = robot.angle_speed
#         # target_distance = robot.task_distance
#         #
#         coord_r = robot.coordinate
#         coord_d = robot.destination
#         target_angle = math.atan2(coord_d[1] - coord_r[1], coord_d[0] - coord_r[0])
#         heading_error = 0
#         if target_angle >= robot.aspect or target_angle <= robot.aspect - math.pi:
#             heading_error = (target_angle - robot.aspect + 2 * math.pi) % math.pi
#         else:
#             heading_error = -((target_angle - robot.aspect + 2 * math.pi) % math.pi)
#
#         # coord_w = robot.dest_wb.coordinate
#         # coord_r = robot.coordinate
#         # aspect = robot.aspect if robot.aspect >= 0 else math.pi - robot.aspect
#         # a = np.array([coord_w[0] - coord_r[0], coord_w[1] - coord_r[1]])
#         # b = np.array([1, 0])
#         # angle = np.arccos(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
#         # big_than_pi = a[0] * b[1] - a[1] * b[0]
#         # angle = 2 * math.pi - angle if big_than_pi > 0 else angle
#
#         # heading_error = math.fabs(angle - aspect) if angle > aspect else -math.fabs(angle - aspect)