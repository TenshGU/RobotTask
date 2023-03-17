class Workbench:
    def __init__(self, ID: int, type_: int, needed: int, cycle: int, X: float, Y: float):
        self.ID = ID
        self.type_ = type_
        self.needed = needed
        self.cycle = cycle
        self.remain = -1
        self.materials = 0b00000000
        self.product = 0
        self.X = X
        self.Y = Y

    def flush_status(self, remain: int, materials: int, product: int):
        self.remain = remain
        self.materials = materials
        self.product = product

    # # if this Workbench need this item of type_ and this type_ not exist in here, return True
    # def judge_suitable(self, type_):
    #     return True if type_ in self.needed and type_ not in self.materials else False
    #
    # # if some Workbench does not need item, it will return False
    # def place_items(self, type_, frame: int) -> bool:
    #     if self.judge_suitable(type_):
    #         self.materials.append(type_)
    #         if len(self.materials) == len(self.needed):
    #             self.begin = frame
    #         return True
    #     return False
    #
    # # whether the robot can take away the product, if robot took away, the self.begin will be reset
    # def take_away(self, frame: int) -> int:
    #     if self.begin >= 0 and frame - self.begin >= self.cycle:
    #         self.begin = -1 if self.needed else frame
    #         return True
    #     return False
    #
    # def remain(self):


class Robot:
    def __init__(self, ID: float, radius: float, X: float, Y: float):
        self.ID = ID
        self.radius = radius
        self.workbench = -1
        self.carry_type = 0
        self.time_coefficient = 0.0
        self.collide_coefficient = 0.0
        self.angle_speed = 0.0
        self.line_speed = 0.0
        self.aspect = 0.0
        self.X = X
        self.Y = Y

    def flush_status(self, workbench: int, carry_type: int, time_coefficient: float, collide_coefficient: float,
              angle_speed: float, line_speed: float, aspect: float, X: float, Y: float):
        self.carry_type = carry_type
        self.time_coefficient = time_coefficient
        self.collide_coefficient = collide_coefficient
        self.angle_speed = angle_speed
        self.line_speed = line_speed
        self.aspect = aspect
        self.X = X
        self.Y = Y

    # def judge_collide(self, Workbenches: []) -> bool:
    #     for workbench in Workbenches:
    #         return False
