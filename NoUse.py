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