import sys

from Object import *

ROBOT_RADIUS = 0.45
NEEDED = [0b00000000, 0b00000000, 0b00000000, 0b00000110, 0b00001010, 0b00001100,
          0b01110000, 0b10000000, 0b11111110]
WORK_CYCLE = [50, 50, 50, 500, 500, 500, 1000, 1, 1]


def read_map(Robots: [], Workbenches: []):
    w_nums, r_nums = 0, 0
    X, Y = 0.25, 49.75
    readline = input()
    while readline != "OK":
        for ch in readline:
            if ch == 'A':
                robot = Robot(r_nums, ROBOT_RADIUS, X, Y)
                Robots.append(robot)
                r_nums += 1
            elif '1' <= ch <= '9':
                type_ = int(ch)
                workbench = Workbench(w_nums, type_, NEEDED[type_], WORK_CYCLE[type_], X, Y)
                Workbenches.append(workbench)
            X += 0.5
        X = 0.25
        Y -= 0.5
        readline = input()


def read_frame(Robots: [], Workbenches: []):
    w_nums, w_len = int(input()), len(Workbenches)
    index = 0
    readline = input()
    while readline != "OK":
        if w_nums > 0:
            parts = readline.split(' ')
            Workbenches[w_nums].flush_status(parts[3], parts[4], parts[5])
            index = index + 1 if index < w_len else 0
            w_nums -= 1
        else:
            Robots[index].flush_status()
            index += 1
        readline = input()


def finish():
    sys.stdout.write('OK\n')
    sys.stdout.flush()
