class Workbench:
    def __init__(self, ID: int, type_: int, needed: [], cycle: int, product: int, X: float, Y: float, frame: int):
        self.ID = ID
        self.type_ = type_
        self.needed = needed
        self.materials = []
        self.begin = -1 if not needed else frame
        self.cycle = cycle
        self.product = product
        self.X = X
        self.Y = Y

    # if this Workbench need this item of type_ and this type_ not exist in here, return True
    def judge_suitable(self, type_):
        return True if type_ in self.needed and type_ not in self.materials else False

    # if some Workbench does not need item, it will return False
    def place_items(self, type_, frame: int) -> bool:
        if self.judge_suitable(type_):
            self.materials.append(type_)
            if len(self.materials) == len(self.needed):
                self.begin = frame
            return True
        return False

    # whether the robot can take away the product, if robot took away, the self.begin will be reset
    def take_away(self, frame: int) -> bool:
        if self.begin >= 0 and frame - self.begin >= self.cycle:
            self.begin = -1 if not self.needed else frame
            return True
        return False


class Robot:
    def __init__(self, radius: float, carry_type: int,
                 time_coefficient: float, collide_coefficient: float,
                 angle_speed: float, line_speed: float, aspect: float,
                 X: float, Y: float):
        self.radius = radius
        self.carry_type = carry_type
        self.time_coefficient = time_coefficient
        self.collide_coefficient = collide_coefficient
        self.angle_speed = angle_speed
        self.line_speed = line_speed
        self.aspect = aspect
        self.X = X
        self.Y = Y

    def flush_coordinate(self, X: float, Y: float):
        self.X = X
        self.Y = Y

    def judge_collide(self, Workbenches: []) -> bool:
        for workbench in Workbenches:
            return False
