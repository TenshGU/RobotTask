#!/bin/bash
import sys
from Object import *
from Util import *

Workbenches = []  # List for Workbench in this map
Robots = [] # List for Robot in this map

# read each frame's main information in here
def read_util_ok():
    a = input()
    while a != "OK":
        a = input()


def finish():
    sys.stdout.write('OK\n')
    sys.stdout.flush()


if __name__ == '__main__':
    # read map
    read_util_ok()
    finish()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.split(' ')
        frame_id = int(parts[0])
        score = int(parts[1])
        read_util_ok()

        # No.frame_id's control
        sys.stdout.write('%d\n' % frame_id)
        line_speed, angle_speed = 3, 1.5
        for robot_id in range(4):
            sys.stdout.write('forward %d %d\n' % (robot_id, line_speed))
            sys.stdout.write('rotate %d %f\n' % (robot_id, angle_speed))
        finish()
