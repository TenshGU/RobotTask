from Object import Robot


ROBOT_RADIUS = 0.45
ROBOT_CARRY_TYPE = 0

def read_map(Robots: [], Workbenches: []):
    w_nums, r_nums = 0, 0
    X, Y = 0.25, 49.75
    readline = input()
    while readline != "OK":
        for ch in readline:
            if ch == 'A':
                robot = Robot(r_nums, ROBOT_RADIUS, ROBOT_CARRY_TYPE,
                              )
                Robots.append(robot)
                r_nums += 1

        readline = input()