import numpy as np
from Object import KDTree


def update_future_value(workbenches: [], waiting_benches: []):
    for wb in waiting_benches:
        wb.update_future_value(workbenches)


# define which workbench should be filtered
def filter_func(robot, wb) -> bool:
    carry_type = robot.carry_type
    needed = wb.needed
    have = wb.materials
    # if robot carry the material that wb needed, and wb haven't
    # if robot doesn't carry any material and wb doesn't need any material
    # here is not consider that whether robot need to destroy material
    return False if ((needed & (1 << carry_type)) > 0 and (have & (1 << carry_type)) == 0) or \
                    (carry_type == 0 and needed == 0) else True


def find_best_workbench(robots: [], workbenches: [], waiting_benches: [], tree: KDTree):
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

    len_r = len(robots)
    selected_w = set()
    selected = {}
    while len(selected_w) == len_r:
        for robot in robots:
            k_nearest = tree.query(robot, k=len_r, filter_func=filter_func)
            for k in k_nearest:
                distance = k[0]
                wb = k[1]
                if wb not in selected_w:
                    selected_w.add(wb)
                    selected[wb] = [robot, distance]
                    break
                else:
                    if selected[wb][1] < distance:
                        continue
                    else:
                        selected[wb] = [robot, distance]

    for key, value in selected.items():
        robot = value[0]
        robot.set_destination(key.coordinate)


def post_operator() -> dict:
    """the operator for the machine should do to move itself(i.e. rotate)"""


def judge_collide() -> bool:
    """judge whether the machines will be collided by themselves"""
