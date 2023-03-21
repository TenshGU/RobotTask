import sys
from Object import *
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


def initialization(robots: [], workbenches: [], type_wbs: {}, have_type: []):
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
                workbench = Workbench(w_nums, type_, const.NEEDED_BIN[type_-1], const.WORK_CYCLE[type_-1],
                                      np.array([X, Y]), const.DIRECT_NEXT[type_-1])
                workbenches.append(workbench)
                w_nums += 1

                wbs = type_wbs.get(type_) if type_wbs.__contains__(type_) else []
                wbs.append(workbench)
                type_wbs.setdefault(type_, wbs)

                if type_ not in have_type:
                    have_type.append(type_)
            X += 0.5
        X = 0.25
        Y -= 0.5

        readline = input()

    # calculate the direct distance
    length = len(workbenches)
    for index_i in range(length):
        workbench_i = workbenches[index_i]
        coordinate_i = workbench_i.coordinate
        for index_j in range(length):
            workbench_j = workbenches[index_j]

            if workbench_j.type_ in const.DIRECT_NEXT[workbench_i.type_-1]:
                coordinate_j = workbench_j.coordinate
                distance = np.linalg.norm(coordinate_i - coordinate_j)
                workbench_i.setup_direct_distance(index_j, distance)
                workbench_j.setup_direct_distance(index_i, distance)


# the main work is to flush the status of wb and robot
def read_frame(robots: [], workbenches: [], waiting_benches: [], type_wbs: {}, have_type: []):
    waiting_benches.clear()
    w_nums = int(input())
    index = 0
    readline = input()
    carrys = set()
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
            carry_type = int(parts[1])
            carrys.add(carry_type)
            index += 1

        readline = input()

    # if robot carries type that the wb need, that wb should be added to waiting_benches
    for carry in carrys:
        if carry == 0:
            continue
        wb_type = const.DIRECT_NEXT[carry-1]
        for type_ in wb_type:
            if type_ in have_type:
                wbs = type_wbs[type_]
                for wb in wbs:
                    if wb not in waiting_benches:
                        waiting_benches.append(wb)


def finish():
    sys.stdout.write('OK\n')
    sys.stdout.flush()
