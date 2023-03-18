import numpy as np


def update_future_value(workbenches: [], waiting_benches: []):
    for wb in waiting_benches:
        wb.update_future_value(workbenches)


def find_best_workbench(robots: [], workbenches: [], waiting_benches: []):
    """
    Dealing with Two-dimensional Nearest Point Pair Problems by Divide and Conquer Method. O(nlogn)
    choose the best workbench for robot to interaction
    current value(the nearest workbench value) + future value(each workbench direct_next value)
    we use greedy policy to make robot to choose which workbench nearest themselves
    after robot get there to finish interaction, the coordinate of workbench will be
    the robot's coordinate, so just move the direct line(which has the best value)
    between w1 and w2, so it reflects the first time, when the robot choose the w1 of the
    best current value, it also contains the future value the robot will take

    we use kd-tree/ball-tree to solve the nearest node finding process
    """
    update_future_value(workbenches, waiting_benches)

    selected_w = set()
    for workbench in workbenches:
        coord_w = workbench.coordinate
        f_v = workbench.future_value
        needed = workbench.needed
        have = workbench.materials
        for robot in robots:
            carry_type = robot.carry_type
            # if robot carry the material that wb needed, and wb haven't
            # if robot doesn't carry any material and wb doesn't need any material
            # here is not consider that whether robot need to destroy material
            if ((needed & (1 << carry_type)) > 0 and (have & (1 << carry_type)) == 0) or \
                    (carry_type == 0 and needed == 0):
                coord_r = robot.coordinate
                value = np.linalg.norm(coord_r - coord_w) + f_v





def post_operator() -> dict:
    """the operator for the machine should do to move itself(i.e. rotate)"""


def judge_collide() -> bool:
    """judge whether the machines will be collided by themselves"""