import sys
from Object import *
import numpy as np
import Constant as const


def initialization(robots: [], workbenches: [], sell_dict: {}, kd_tree: []):
    w_nums, r_nums = 0, 0
    X, Y = 0.25, 49.75
    readline = input()
    while readline != "OK":
        for ch in readline:
            if ch == 'A':
                robot = Robot(r_nums, const.ROBOT_RADIUS, np.array([X, Y]))
                robots.append(robot)
                r_nums += 1
            elif '1' <= ch <= '9':
                type_ = int(ch)
                workbench = Workbench(w_nums, type_, const.NEEDED_BIN[type_], const.WORK_CYCLE[type_],
                                      np.array([X, Y]), const.DIRECT_NEXT[type_])
                workbenches.append(workbench)

                if type_ >= 4:
                    for i in const.NEEDED_DEC[type_]:
                        coordinate = sell_dict.get(i) if sell_dict.__contains__(i) else []
                        coordinate.append(np.array([X, Y]))
                        sell_dict.setdefault(type_, coordinate)
            X += 0.5
        X = 0.25
        Y -= 0.5

        readline = input()

    # calculate the direct distance
    length = len(workbenches)
    for index_i in range(length):
        workbench_i = workbenches[index_i]
        coordinate_i = workbench_i.coordinate
        for index_j in range(index_i + 1, length):
            workbench_j = workbenches[index_j]

            if workbench_j.type_ in const.DIRECT_NEXT[workbench_j.type_]:
                coordinate_j = workbench_j.coordinate
                distance = np.linalg.norm(coordinate_i - coordinate_j)
                workbench_i.setup_direct_distance(index_j, distance)
                workbench_j.setup_direct_distance(index_i, distance)
        index_i += 1

    # construct the kd-tree
    pointset = np.empty((0, 2))
    workbenches_np = np.array(workbenches)
    for workbench in workbenches:
        point = workbench.coordinate
        pointset = np.vstack([pointset, point])
    tree = KDTree(pointset, workbenches_np)
    kd_tree.append(tree)


# the main work is to flush the status of wb and robot
def read_frame(robots: [], workbenches: [], waiting_benches: []):
    waiting_benches.clear()
    w_nums = int(input())
    index = 0
    readline = input()
    while readline != "OK":
        parts = readline.split(' ')
        if w_nums > 0:
            workbenches[index].flush_status(int(parts[3]), int(parts[4]), int(parts[5]))
            # add the wb which has product can be taken way
            if int(parts[5]) == 1:
                waiting_benches.append(workbenches[index])
            index = (index + 1) if w_nums - 1 > 0 else 0
            w_nums -= 1
        else:
            robots[index].flush_status(int(parts[0]), int(parts[1]), float(parts[2]),
                                       float(parts[3]), float(parts[4]), float(parts[5]),
                                       float(parts[6]), float(parts[7]),
                                       np.array([float(parts[8]), float(parts[9])]))
            index += 1

        readline = input()


def finish():
    sys.stdout.write('OK\n')
    sys.stdout.flush()
