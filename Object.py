from math import floor
import numpy as np
import Constant as const
import logging
logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)
# 建立一个filehandler来把日志记录在文件里，级别为debug以上
fh = logging.FileHandler("spam.log")
fh.setLevel(logging.DEBUG)
# 建立一个streamhandler来把日志打在CMD窗口上，级别为error以上
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
#将相应的handler添加在logger对象中
logger.addHandler(ch)
logger.addHandler(fh)


class Workbench:
    def __init__(self, ID: int, type_: int, angle: float, profit: int, needed: int, cycle: int, coordinate: np.ndarray, direct_next: []):
        self.ID = ID
        self.type_ = type_
        self.angle = angle
        self.profit = profit
        self.needed = needed
        self.cycle = cycle
        self.product_type = -1 if type_ > 7 else type_
        # will be flush by system
        self.remain = -1
        self.materials = 0
        self.product = 0
        # fixed
        self.coordinate = coordinate
        self.direct_next = direct_next
        self.direct_distance = {}  # distance for each direct workbench, key:ID in wbs, value:distance
        # should be updated, when robot find the best wb
        self.future_value = float('inf')  # the best future value
        self.future_next = None

    def flush_status(self, remain: int, materials: int, product: int):
        self.remain = remain
        self.materials = materials
        self.product = product

    def setup_direct_distance(self, ID: int, distance: float):
        self.direct_distance[ID] = distance

    def update_future_value(self, workbench: []):
        min_value = float('inf')
        for key, value in self.direct_distance.items():
            if min_value > value:
                min_value = value
                self.future_next = workbench[key]
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

        self.pre_angle_speed = 0
        self.pre_line_speed_x = 0
        self.pre_line_speed_y = 0
        self.expect_type = 0
        self.destination = np.array([-2, -2])
        self.task_distance = 0
        self.product_value = 0
        self.urgency = -1
        self.last_interaction = np.array([-1, -1])
        self.dest_wb = None

    def flush_status(self, workbench: int, carry_type: int, time_coefficient: float, collide_coefficient: float,
                     angle_speed: float, line_speed_x: float, line_speed_y: float, aspect: float, coordinate: np.ndarray):
        self.workbench = workbench
        self.carry_type = carry_type
        self.time_coefficient = time_coefficient
        self.collide_coefficient = collide_coefficient
        self.pre_angle_speed = self.angle_speed
        self.pre_line_speed_x = self.pre_line_speed_x
        self.pre_line_speed_y = self.pre_line_speed_y
        self.angle_speed = angle_speed
        self.line_speed_x = line_speed_x
        self.line_speed_y = line_speed_y
        self.aspect = aspect
        self.coordinate = coordinate
        if carry_type != 0:
            self.product_value = floor(const.PRICE[carry_type-1][1] * time_coefficient * collide_coefficient)
            self.radius = const.ROBOT_RADIUS_PRODUCT
        else:
            self.radius = const.ROBOT_RADIUS
            # reset the attribute not belong to the system given
            self.destination = np.array([-1, -1])
            self.task_distance = 0
            self.product_value = 0
