import numpy as np


class Workbench:
    def __init__(self, ID: int, type_: int, needed: int, cycle: int, coordinate: np.ndarray, direct_next: []):
        self.ID = ID
        self.type_ = type_
        self.needed = needed
        self.cycle = cycle
        self.remain = -1
        self.materials = 0b00000000
        self.product = 0
        self.coordinate = coordinate
        self.direct_next = direct_next
        self.direct_distance = {}  # distance for each direct workbench, key:ID, value:distance
        self.future_value = 0  # the best future value

    def flush_status(self, remain: int, materials: int, product: int):
        self.remain = remain
        self.materials = bin(materials)
        self.product = product

    def setup_direct_distance(self, ID: int, distance: float):
        self.direct_distance.setdefault(ID, distance)

    def update_future_value(self, workbenches: []):
        min_value = float('inf')
        for key, value in self.direct_distance.items():
            next_remain = workbenches[key].__dict__['remain']
            if next_remain != -1:
                current = value + next_remain
                min_value = current if min_value > current else min_value
        self.future_value = min_value


class Robot:
    def __init__(self, ID: float, radius: float, coordinate: np.ndarray):
        self.ID = ID
        self.radius = radius
        self.workbench = -1
        self.carry_type = 0
        self.time_coefficient = 0.0
        self.collide_coefficient = 0.0
        self.angle_speed = 0.0
        self.line_speed_x = 0.0
        self.line_speed_y = 0.0
        self.aspect = 0.0
        self.coordinate = coordinate
        self.destination = np.array([-1, -1])

    def flush_status(self, workbench: int, carry_type: int, time_coefficient: float, collide_coefficient: float,
                     angle_speed: float, line_speed_x: float, line_speed_y: float, aspect: float, coordinate: np.ndarray):
        self.workbench = workbench
        self.carry_type = carry_type
        self.time_coefficient = time_coefficient
        self.collide_coefficient = collide_coefficient
        self.angle_speed = angle_speed
        self.line_speed_x = line_speed_x
        self.line_speed_y = line_speed_y
        self.aspect = aspect
        self.coordinate = coordinate

    def set_destination(self, destination_: np.ndarray):
        self.destination = destination_


class KDTree:
    def __init__(self, data):
        self.k = data.shape[1]  # 数据维度
        self.tree = self.build(data)

    class Node:
        def __init__(self, data, left, right):
            self.data = data
            self.left = left
            self.right = right

    def build(self, data, depth=0):
        if len(data) == 0:
            return None
        axis = depth % self.k
        sorted_idx = np.argsort(data[:, axis])
        data_sorted = data[sorted_idx]
        mid = len(data) // 2
        return self.Node(data_sorted[mid],
                         self.build(data_sorted[:mid], depth + 1),
                         self.build(data_sorted[mid + 1:], depth + 1))

    def query(self, x, k=1):
        def search_knn(node, x, k, heap):
            if node is None:
                return
            dist = np.linalg.norm(node.data - x)
            if len(heap) < k:
                heap.append((dist, node.data))
            elif dist < heap[-1][0]:
                heap[-1] = (dist, node.data)
            axis = node.data.argmax()  # 取最大值对应的维度
            if x[axis] < node.data[axis]:
                search_knn(node.left, x, k, heap)
            else:
                search_knn(node.right, x, k, heap)

        heap = []
        search_knn(self.tree, x, k, heap)
        return sorted(heap)
