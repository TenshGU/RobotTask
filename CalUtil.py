import heapq
import math
import numpy as np
import Constant as const
from DWA import Info, DWA_Core
from Object import Robot, Workbench


def pre_operator(workbenches: []):
    filtered = [wb for wb in workbenches if wb.remain >= 0]
    k = len(filtered)
    return heapq.nsmallest(k, filtered, key=lambda wb: wb.remain)


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
    filtered = [wb for wb in waiting_benches if not filter_func(obj_x, wb, remain_count, robot_carry_type)]
    result_list = heapq.nsmallest(k, filtered, key=lambda wb: sort_rule(wb, robot=obj_x))
    result_with_distance = [[np.linalg.norm(wb.coordinate - obj_x.coordinate) + wb.future_value, wb] for wb in
                            result_list]
    return result_with_distance


def update_future_value(workbenches: [], waiting_benches: []):
    for wb in waiting_benches:
        wb.update_future_value(workbenches)


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
        key.lock = True


def post_operator(robots: []) -> []:
    """the operator for the machine should do to move itself(i.e. rotate)"""
    res = []
    for robot in robots:
        opt_dict = {}
        opt_dict_ = {'forward': 0, 'rotate': 0, 'buy': -1, 'sell': -1}
        if robot.dest_wb:
            buy = cal_buy(robot)
            sell = cal_sell(robot)

            line_speed, angle_speed = 0, 0
            collide, r2 = maybe_collide(robot, robots)
            if collide:
                obstacles_list = [rot.coordinate for rot in robots if rot != robot]
                obstacles = np.array(obstacles_list)
                info = Info(robot, obstacles)
                u = DWA_Core(info)
                line_speed = u[0]
                angle_speed = u[1]
            else:
                coord_r = robot.coordinate
                coord_d = robot.destination
                target_angle = math.atan2(coord_d[1] - coord_r[1], coord_d[0] - coord_r[0])
                heading_error = 0
                if ((robot.aspect < 0 and target_angle > 0) or (robot.aspect > 0 and target_angle < 0)) \
                        and math.fabs(math.fabs((target_angle - robot.aspect) - math.pi)) <= 0.1:
                    heading_error = math.pi
                else:
                    if target_angle >= robot.aspect or target_angle <= robot.aspect - math.pi:
                        heading_error = (target_angle - robot.aspect + 2 * math.pi) % math.pi
                    else:
                        heading_error = -((target_angle - robot.aspect + 2 * math.pi) % math.pi)

                angle = math.fabs(heading_error)
                line_speed, angle_speed = 6, 2

                if robot.carry_type == 0:
                    if buy != -1:
                        line_speed, angle_speed = 0, 0
                    else:
                        if angle <= 0.1:
                            angle_speed = 0
                            if robot.task_distance < 1.5:
                                line_speed = 1
                        else:
                            line_speed = 1
                else:
                    if sell != -1:
                        line_speed, angle_speed, = 0, 0
                    else:
                        if angle <= 0.1:
                            angle_speed = 0
                            if robot.task_distance < 1.5:
                                line_speed = 1
                        else:
                            line_speed = 1
                angle_speed = angle_speed if heading_error >= 0 else -angle_speed
            opt_dict['forward'] = line_speed
            opt_dict['rotate'] = angle_speed
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


def judge_collide(robot: Robot, robots: []) -> bool:
    """
    judge whether the machines will be collided by themselves
    if robot will be collided, choose a lower proprity one to slow down/rotate
    if their proprity equals, random to choose one to slow down/rotate
    here only consider the robot slow down
    """
    for rot in robots:
        if rot == robot:
            continue
        current_radius = const.ROBOT_RADIUS if robot.carry_type == 0 else const.ROBOT_RADIUS_PRODUCT
        other_radius = const.ROBOT_RADIUS if rot.carry_type == 0 else const.ROBOT_RADIUS_PRODUCT
        collide_range = current_radius + other_radius
        distance = np.linalg.norm(rot.coordinate - robot.coordinate)
        if maybe_collide(robot, rot) and distance <= collide_range + const.COLLIDE_DISTANCE:
            return True
    return False


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
        robot.dest_wb.lock = False
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
        robot.dest_wb.lock = False
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
        beta = np.arccos((dx / distance))
    if dy < 0:
        beta = 2 * math.pi - beta
    if robot.aspect < 0:
        alfa = 2 * math.pi + robot.aspect
    return 0 if math.fabs(beta - alfa) < 0.01 else math.fabs(beta - alfa)


def maybe_collide(r1: Robot, robots: []):
    for r2 in robots:
        if r2 == r1:
            continue
        if np.linalg.norm(r1.coordinate - r2.coordinate) >= r1.radius + r2.radius + const.COLLIDE_DISTANCE:
            continue
        if math.fabs(math.fabs(r1.aspect - r2.aspect) - math.pi) <= 0.01 or math.fabs(r1.aspect - r2.aspect) <= 0.01:
            return True, r2
        else:
            x0, y0 = -1, -1
            if math.fabs(r1.aspect - (math.pi / 2)) <= 0.01 and math.fabs(r2.aspect - (math.pi / 2)) >= 0.01:
                x0 = r1.coordinate[0]
                point2 = r2.coordinate
                asp2 = r2.aspect if r2.aspect >= 0 else math.pi + r2.aspect
                k2 = math.tan(asp2)
                b2 = point2[1] - k2 * point2[0]
                y0 = k2 * x0 + b2
            elif math.fabs(r2.aspect - (math.pi / 2)) <= 0.01 and math.fabs(r1.aspect - (math.pi / 2)) >= 0.01:
                x0 = r2.coordinate[0]
                point1 = r1.coordinate
                asp1 = r1.aspect if r1.aspect >= 0 else math.pi + r1.aspect
                k1 = math.tan(asp1)
                b1 = point1[1] - k1 * point1[0]
                y0 = k1 * x0 + b1
            else:
                asp1 = r1.aspect if r1.aspect >= 0 else math.pi + r1.aspect
                asp2 = r2.aspect if r2.aspect >= 0 else math.pi + r2.aspect
                k1 = math.tan(asp1)
                k2 = math.tan(asp2)
                point1 = r1.coordinate
                point2 = r2.coordinate
                b1 = point1[1] - k1 * point1[0]
                b2 = point2[1] - k2 * point2[0]

                x0 = (b2 - b1) / (k1 - k2)
                y0 = k1 * x0 + b1

            if x0 > 0 and y0 > 0:
                angle1 = math.atan2(y0 - r1.coordinate[1], x0 - r1.coordinate[0])
                angle2 = math.atan2(y0 - r2.coordinate[1], x0 - r2.coordinate[0])
                point_side1 = True if math.fabs(angle1 - r1.aspect) <= 0.01 else False
                point_side2 = True if math.fabs(angle2 - r2.aspect) <= 0.01 else False
                if point_side1 and point_side2:
                    return True, r2
    return False, None
