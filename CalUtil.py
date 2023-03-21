import array
import heapq
import math
import numpy as np
import Constant as const
from Object import Robot
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


def space_search(waiting_benches: [], obj_x, k=1, filter_func=None) -> []:
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
    filtered = [wb for wb in waiting_benches if not filter_func(obj_x, wb)]
    result_list = heapq.nsmallest(k, filtered,
                                  key=lambda wb: (np.linalg.norm(wb.coordinate - obj_x.coordinate) + wb.future_value))
    result_with_distance = [[np.linalg.norm(wb.coordinate - obj_x.coordinate) + wb.future_value, wb] for wb in
                            result_list]
    return result_with_distance


def pre_operator(workbenches: []):
    filtered = [wb for wb in workbenches if wb.remain >= 0]
    k = len(filtered)
    return heapq.nsmallest(k, filtered, key=lambda wb: wb.remain)


def update_future_value(waiting_benches: []):
    for wb in waiting_benches:
        wb.update_future_value()


# filter which wb that robot can get there
def filter_func(robot, wb) -> bool:
    carry_type = robot.carry_type
    needed = wb.needed
    have = wb.materials
    product = wb.product
    # if robot carry the material that wb needed, and wb haven't(can sell 4-9)
    # if robot doesn't carry any material and wb doesn't need any material(can go to 1,2,3)
    # if robot doesn't carry any material and wb product(can buy 1-7)
    # here is not consider that whether robot need to destroy material
    return False if ((needed & (1 << carry_type)) > 0 and (have & (1 << carry_type)) == 0) or \
                    (carry_type == 0 and (product == 1 or needed == 0)) or \
                    (not (robot.last_interaction == robot.destination).all) else True


def find_best_workbench(robots: [], waiting_benches: []):
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
    update_future_value(waiting_benches)

    k_nearest_dict = {}
    selected_w = set()
    selected_r = set()
    selected = {}
    lock = []

    max_k = -1

    for robot in robots:
        k_nearest = space_search(waiting_benches, robot, k=const.ROBOT_NUM, filter_func=filter_func)
        k_nearest_dict[robot] = k_nearest
        max_k = len(k_nearest) if len(k_nearest) > max_k else max_k

    choose_num = const.ROBOT_NUM if len(waiting_benches) > const.ROBOT_NUM else len(waiting_benches)

    # if all robot has chosen the wb(it just means the number of wb was enough), the loop will break
    while len(selected_r) != max_k:
        for robot in robots:
            no_choose = True
            # if robot has select a wb, or current robot has not grabbed by other robot
            if robot in selected_r:
                continue
            k_nearest = k_nearest_dict[robot]
            # choosing the best wb
            for k in k_nearest:
                distance, wb = k[0], k[1]
                # if this wb has not grabbed by other robot, choose it
                if wb not in selected_w:
                    selected[wb] = [robot, distance]
                    selected_w.add(wb)
                    selected_r.add(robot)
                    no_choose = False
                    break
                # if this wb has grabbed by other robot, judge whether this robot distance less than pre-choose distance
                # if more than it, continue to choose the next wb
                # if less than, grab it, the pre-choose one will be removed from the selected_r
                # after that, begin to the next robot
                else:
                    old_robot = selected[wb][0]
                    if old_robot not in lock:
                        if distance >= selected[wb][1]:
                            continue
                        else:
                            selected[wb] = [robot, distance]
                            selected_r.remove(old_robot)
                            selected_r.add(robot)
                            no_choose = False
                            break
            # force to choose the first one, and lock it
            if no_choose:
                k = k_nearest[0]
                distance, wb = k[0], k[1]
                old_robot = selected[wb][0]
                selected[wb] = [robot, distance]
                selected_r.remove(old_robot)
                selected_r.add(robot)
                lock.append(robot)

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
        l_speed = math.sqrt(robot.line_speed_x ** 2 + robot.line_speed_y ** 2)
        a_speed = robot.angle_speed
        target_direction = cal_angle(robot)
        target_distance = robot.task_distance

        line_speed, angle_speed = calculate_next_velocity(l_speed, a_speed, target_direction, target_distance)

        heading_error = target_direction - robot.aspect
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        elif heading_error < -math.pi:
            heading_error += 2 * math.pi

        opt_dict['forward'] = cal_robots_forward(robot)
        opt_dict['rotate'] = min((heading_error * 50), math.pi) if heading_error != 0 else 0
        opt_dict['buy'] = cal_buy(robot)
        opt_dict['sell'] = cal_sell(robot)
        res.append(opt_dict)
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


def cal_robots_forward(robot: Robot) -> int:
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
    return 6


def cal_angle(robot: Robot) -> float:
    coord_r = robot.coordinate
    coord_d = robot.destination
    angle = math.atan2(coord_d[1] - coord_r[1], coord_d[0] - coord_r[0])
    return angle


def cal_buy(robot: Robot) -> int:
    # judge whether the robot has interacted with wb, or the robot will stay in here to take repeat action
    # based on the robot will never interact with the same wb in near two times
    allow_interact = True if robot.dest_wb.product == 1 else False
    if ((not (robot.last_interaction == robot.destination).all) or (robot.last_interaction == np.array([-1, -1])).all) \
            and allow_interact:
        robot.last_interaction = robot.destination
        if robot.carry_type == 0 and robot.task_distance < const.INTERACTION_RADIUS:
            return robot.expect_type
    return -1


def cal_sell(robot: Robot) -> int:
    # judge whether the robot has interacted with wb, or the robot will stay in here to take repeat action
    # based on the robot will never interact with the same wb in near two times
    carry_type = robot.carry_type
    needed = robot.dest_wb.needed
    have = robot.dest_wb.materials
    allow_interact = True if ((needed & (1 << carry_type)) > 0 and (have & (1 << carry_type)) == 0) else False
    if ((not (robot.last_interaction == robot.destination).all) or (robot.last_interaction == np.array([-1, -1])).all) \
            and allow_interact:
        robot.last_interaction = robot.destination
        if robot.carry_type != 0 and robot.task_distance < const.INTERACTION_RADIUS:
            return robot.carry_type
    return -1


def calculate_next_velocity(current_speed, current_heading, target_direction, target_distance):
    # 计算角速度
    heading_error = target_direction - current_heading
    if heading_error > math.pi:
        heading_error -= 2 * math.pi
    elif heading_error < -math.pi:
        heading_error += 2 * math.pi
    max_torque = 50
    angular_acceleration = heading_error * max_torque / 20  # 角加速度 = 角误差 * 最大力矩 / (密度 * 半径^2)
    max_angular_speed = math.pi
    if angular_acceleration > 0:
        max_angular_speed = min(max_angular_speed, math.sqrt(2 * angular_acceleration * math.pi / 50))
    else:
        max_angular_speed = max(-max_angular_speed, -math.sqrt(2 * abs(angular_acceleration) * math.pi / 50))
    angular_speed = max(min(max_angular_speed, math.pi), -math.pi)

    # 计算线速度
    max_force = 250
    max_speed_forward = 6.0
    max_speed_backward = -2.0
    speed_error = min(max_speed_forward, max_speed_backward, target_distance) - current_speed
    linear_acceleration = speed_error * max_force / 20  # 线加速度 = 速度误差 * 最大牵引力 / 密度
    if linear_acceleration > 0:
        max_linear_speed = min(max_speed_forward, math.sqrt(2 * linear_acceleration * 6))
    else:
        max_linear_speed = max(max_speed_backward, -math.sqrt(2 * abs(linear_acceleration) * 2))
    linear_speed = max(min(max_linear_speed, 6), -2)

    return angular_speed, linear_speed
