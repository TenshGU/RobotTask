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
