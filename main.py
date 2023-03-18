#!/bin/bash
import sys
import InfoUtil

robots = []  # List for Robot in this map
workbenches = []  # List for Workbench in this map
sell_dict = {}  # key: type  value: coordinate {1:[[xx,yy], ...]} the coordinate of workbench that bobot can sell type
waiting_benches = []  # when read frame, full it with which wbs product can be taken
price = [[3000, 6000], [4400, 7600], [5800, 9200], [15400, 22500],
         [17200, 25000], [19200, 27500], [76000, 105000]]  # price for each product


# read each frame's main information in here
def read_util_ok():
    a = input()
    while a != "OK":
        a = input()


def finish():
    sys.stdout.write('OK\n')
    sys.stdout.flush()


if __name__ == '__main__':
    # read map and initialization
    InfoUtil.initialization(robots, workbenches, sell_dict)
    InfoUtil.finish()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.split(' ')
        frame_id = int(parts[0])
        score = int(parts[1])
        InfoUtil.read_frame(robots, workbenches, waiting_benches)

        # main algorithm

        # No.frame_id's control
        sys.stdout.write('%d\n' % frame_id)
        line_speed, angle_speed = 3, 1.5
        for robot_id in range(4):
            sys.stdout.write('forward %d %d\n' % (robot_id, line_speed))
            sys.stdout.write('rotate %d %f\n' % (robot_id, angle_speed))
        InfoUtil.finish()
