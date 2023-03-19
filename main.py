#!/bin/bash
from InfoUtil import *
from CalUtil import find_best_workbench, post_operator, pre_operator
import Constant as const

robots = []  # List for Robot in this map
workbenches = []  # List for Workbench in this map
sell_dict = {}  # key: type  value: coordinate {1:[[xx,yy], ...]} the coordinate of workbench that bobot can sell type
waiting_benches = []  # when read frame, full it with which wbs product can be taken
kd_tree = []  # KD-Tree


if __name__ == '__main__':
    # read map and initialization
    initialization(robots, workbenches, sell_dict, kd_tree)
    tree = kd_tree[0]
    finish()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.split(' ')
        frame_id = int(parts[0])
        score = int(parts[1])
        read_frame(robots, workbenches, waiting_benches)

        # wb have not any product can be taken away, choose the wb which has less remain
        if not waiting_benches:
            waiting_benches = pre_operator(workbenches)
        # main algorithm
        find_best_workbench(robots, workbenches, waiting_benches, tree)
        operation = post_operator(robots)

        # No.frame_id's control
        sys.stdout.write('%d\n' % frame_id)
        for i in range(const.ROBOT_NUM):
            opt_dict = operation[i]
            for key, value in opt_dict.items():
                if key == 'forward':
                    sys.stdout.write('forward %d %d\n' % (i, value))
                elif key == 'rotate':
                    sys.stdout.write('rotate %d %f\n' % (i, value))
                elif key == 'buy':
                    if value == -1:
                        continue
                    else:
                        sys.stdout.write('buy %d\n' % i)
                elif key == 'sell':
                    if value == -1:
                        continue
                    else:
                        sys.stdout.write('sell %d\n' % i)
        finish()
