import array
import heapq
import math
import numpy as np
import Constant as const
from Object import Robot, Workbench
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


def sort_rule(wb: Workbench, robot: Robot):
    distance = np.linalg.norm(wb.coordinate - robot.coordinate)
    type_ = wb.type_
    needed = wb.future_next.needed
    have = wb.future_next.materials
    p = 0
    if (needed & (1 << type_)) > 0 and (have & (1 << type_)) == 0:
        p = wb.profit / (distance + wb.future_value)  # have meaning
    return distance + wb.future_value - p


def space_search(waiting_benches: [], remain_count: {}, robot_carry_type: [], obj_x, k=1, filter_func=None) -> []:
    # heap = []
    # never_selected = []
    # for wb in waiting_benches:
    #     dist = np.linalg.norm(wb.coordinate - obj_x.coordinate) + wb.future_value
    #     if not filter_func(obj_x, wb):
    #         if len(heap) < k:
    #             heap.append((dist, wb))
    #         elif dist < heap[-1][0]:
    #             heap[-1] = (dist, wb)
    #         if dist < heap[]
    #     else:
    #         never_selected.append([dist, wb])
    # # if not enough k, full it with never_selected, it looks like to disperse themselves
    # if len(heap) < k:
    #     dis = k - len(heap)
    #     for i in range(dis):
    #         heap.append([never_selected[i][0], never_selected[i][1]])
    # logger.info(len(heap))
    # return sorted(heap, key=lambda res: res[0])
    filtered = [wb for wb in waiting_benches if not filter_func(obj_x, wb, remain_count, robot_carry_type)]
    result_list = heapq.nsmallest(k, filtered, key=lambda wb: sort_rule(wb, robot=obj_x))
    result_with_distance = [[np.linalg.norm(wb.coordinate - obj_x.coordinate) + wb.future_value, wb] for wb in
                            result_list]
    return result_with_distance


def pre_operator(workbenches: []):
    filtered = [wb for wb in workbenches if wb.remain >= 0]
    k = len(filtered)
    return heapq.nsmallest(k, filtered, key=lambda wb: wb.remain)


def update_future_value(workbenches: [], waiting_benches: []):
    for wb in waiting_benches:
        wb.update_future_value(workbenches)


# filter which wb that robot can get there
def filter_func(robot, wb, remain_count: {}, robot_carry_type: []) -> bool:
    carry_type = robot.carry_type
    type_ = wb.type_
    needed = wb.needed
    have = wb.materials
    product = wb.product
    count = remain_count[type_] if remain_count.__contains__(type_) else 0
    for rct in robot_carry_type:
        if rct == type_:
            count -= 1
    # if robot carry the material that wb needed, and wb haven't(can sell 4-9)
    # if robot doesn't carry any material and wb doesn't need any material(can go to 1,2,3)
    # if robot doesn't carry any material and wb product(can buy 1-7)
    # here is not consider that whether robot need to destroy material
    can_choose = True if count >= 1 else False
    can_sell = True if (needed & (1 << carry_type)) > 0 and (have & (1 << carry_type)) == 0 else False
    can_buy_go = True if carry_type == 0 and (product == 1 or needed == 0) else False
    not_last_interaction = True if not (robot.last_interaction == robot.destination).all else False
    return False if can_choose and (can_sell or can_buy_go or not_last_interaction) else True


def find_best_workbench(robots: [], workbenches: [], waiting_benches: [], remain_count: {}):
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

    k_nearest_list = []
    selected_w = set()
    selected_r = set()
    selected = {}

    robot_carry_type = [robot.carry_type for robot in robots]

    for robot in robots:
        k_nearest = space_search(waiting_benches, remain_count, robot_carry_type, robot,
                                 k=2 * const.ROBOT_NUM, filter_func=filter_func)
        k_nearest_list.append(k_nearest)
    sorted_indexes = [i for i, _ in sorted(enumerate(k_nearest_list), key=lambda k_nearest: len(k_nearest))]

    for index in sorted_indexes:
        robot = robots[index]
        k_nearest = k_nearest_list[index]
        # choosing the wb
        for k in k_nearest:
            distance, wb = k[0], k[1]
            # if this wb has not grabbed by other robot, choose it
            if wb not in selected_w:
                selected[wb] = [robot, distance]
                selected_w.add(wb)
                selected_r.add(robot)
                break
            else:
                continue

    for key, value in selected.items():
        robot = value[0]
        robot.destination = key.coordinate
        robot.task_distance = np.linalg.norm(robot.coordinate - robot.destination)
        robot.expect_type = key.product_type
        robot.dest_wb = key


def post_operator(robots: []) -> []:
    """the operator for the machine should do to move itself(i.e. rotate)"""
    setup_urgency(robots)
    res = []
    for robot in robots:
        opt_dict = {}
        opt_dict_ = {'forward': 6, 'rotate': 1.5, 'buy': -1, 'sell': -1}
        if robot.dest_wb:
            coord_r = robot.coordinate
            coord_d = robot.destination
            target_angle = math.atan2(coord_d[1] - coord_r[1], coord_d[0] - coord_r[0])
            heading_error = 0
            if target_angle >= robot.aspect or target_angle <= robot.aspect - math.pi:
                heading_error = (target_angle - robot.aspect + 2 * math.pi) % math.pi
            else:
                heading_error = -((target_angle - robot.aspect + 2 * math.pi) % math.pi)

            buy = cal_buy(robot)
            sell = cal_sell(robot)
            angle = rotate_angle(robot)
            line_speed, angle_speed = 6, 1.5

            if robot.carry_type == 0:
                if buy != -1:
                    line_speed, angle_speed = 0, 0
                else:
                    if angle <= 0.1:
                        angle_speed = 0
                        if robot.task_distance < 1:
                            line_speed = 1
                    else:
                        line_speed = 1
            else:
                if sell != -1:
                    line_speed, angle_speed, = 0, 0
                else:
                    if angle <= 0.1:
                        angle_speed = 0
                        if robot.task_distance < 1:
                            line_speed = 1
                    else:
                        line_speed = 1

            opt_dict['forward'] = line_speed
            opt_dict['rotate'] = angle_speed if heading_error >= 0 else -angle_speed
            opt_dict['buy'] = buy
            opt_dict['sell'] = sell
            res.append(opt_dict)
        else:
            res.append(opt_dict_)
    return res


def setup_urgency(robots: []):
    all_distance = 0
    max_value = const.ALL_VALUE
    for robot in robots:
        if robot.product_value == 0:
            all_distance += robot.task_distance
    upper = all_distance + max_value
    for robot in robots:
        if robot.product_value == 0:
            robot.urgency = const.LOWEST_PRIORITY  # this robot has not taken any material
        robot.urgency = (upper - (robot.task_distance + robot.product_value)) / upper


def judge_collide(robots: []) -> bool:
    """
    judge whether the machines will be collided by themselves
    if robot will be collided, choose a lower proprity one to slow down/rotate
    if their proprity equals, random to choose one to slow down/rotate
    here only consider the robot slow down
    """


def cal_forward(robot: Robot, angle: float) -> float:
    # sorted_indexes = [i for i, _ in sorted(enumerate(robots), key=lambda i_r: i_r[1].urgency, reverse=True)]
    # max_index = sorted_indexes[0]
    # arr = array.array('f', const.INIT_SPEED)
    # arr[max_index] = const.ROBOT_MAX_SPEED
    # for index in sorted_indexes[1:]:
    #     """1"""
    # return list(arr)
    # if robot.task_distance > const.INTERACTION_RADIUS + 0.5:
    #     return 6
    # else:
    #     return 0
    if robot.task_distance < 1.5:
        # l_speed = math.sqrt(robot.line_speed_x ** 2 + robot.line_speed_y ** 2)
        return 1
    else:
        if angle >= 0.1:
            return 2
        else:
            return 6  # (-24 / math.pi) * heading_error + 6


def cal_buy(robot: Robot) -> int:
    # judge whether the robot has interacted with wb, or the robot will stay in here to take repeat action
    # based on the robot will never interact with the same wb in near two times
    not_last_interaction = False
    allow_buy = False
    # the wb should not be the last interact wb
    if (not (robot.last_interaction == robot.destination).all) or (robot.last_interaction == np.array([-1, -1])).all:
        not_last_interaction = True
    # only when the robot in interaction radius and the product can be taken away
    if robot.dest_wb.product == 1 and robot.carry_type == 0 and robot.task_distance < const.INTERACTION_RADIUS:
        allow_buy = True
    if not_last_interaction and allow_buy:
        robot.last_interaction = robot.destination
        return robot.expect_type
    return -1


def cal_sell(robot: Robot) -> int:
    # judge whether the robot has interacted with wb, or the robot will stay in here to take repeat action
    # based on the robot will never interact with the same wb in near two times
    carry_type = robot.carry_type
    needed = robot.dest_wb.needed
    have = robot.dest_wb.materials
    not_last_interaction = False
    allow_sell = False
    if (not (robot.last_interaction == robot.destination).all) or (robot.last_interaction == np.array([-1, -1])).all:
        not_last_interaction = True
    if ((needed & (1 << carry_type)) > 0 and (have & (1 << carry_type)) == 0) \
            and robot.carry_type != 0 and robot.task_distance < const.INTERACTION_RADIUS:
        allow_sell = True
    if not_last_interaction and allow_sell:
        robot.last_interaction = robot.destination
        return robot.carry_type
    return -1


def rotate_angle(robot: Robot):
    alfa = robot.aspect
    distance = robot.task_distance
    coord_w = robot.dest_wb.coordinate
    coord_r = robot.coordinate
    dx = coord_w[0] - coord_r[0]
    dy = coord_w[1] - coord_r[1]

    beta = 0
    if distance > 0:
        beta = np.arccos(dx / distance)
    if dy < 0:
        beta = 2 * math.pi - beta
    if robot.aspect < 0:
        alfa = 2 * math.pi + robot.aspect
    return 0 if math.fabs(beta - alfa) < 0.01 else math.fabs(beta - alfa)
