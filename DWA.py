import numpy as np
from math import *
from Object import Robot
import Constant as const


class Info:
    def __init__(self, robot: Robot, obstacles: np.ndarray):
        self.v_min = -2.0 / 50  # 最小速度
        self.v_max = 6.0 / 50  # 最大速度
        self.w_max = pi / 50  # 最大角速度
        self.w_min = -pi / 50  # 最小角速度
        self.vacc_max = (const.FORCE / (const.ROU * pi * (robot.radius ** 2))) / 50
        self.wacc_max = (const.M / (const.ROU * pi * (robot.radius ** 4))) / 50
        # self.vacc_max = sqrt(robot.line_speed_x ** 2 + robot.line_speed_y ** 2) - sqrt(robot.pre_line_speed_x ** 2 +
        #                                                                            robot.pre_line_speed_y ** 2)  # 加速度
        # self.wacc_max = robot.angle_speed - robot.pre_angle_speed  # 角加速度
        self.radius = robot.radius  # 机器人模型半径
        self.x = np.array([robot.coordinate[0], robot.coordinate[1], robot.aspect,
                           sqrt(robot.line_speed_x ** 2 + robot.line_speed_y ** 2) / 50, robot.angle_speed / 50])
        self.goal = robot.destination
        self.obstacles = obstacles  # other robots coordinate

        self.dt = 1 * 15  # 单位为20ms(20 * 1帧)，每次的运动轨迹是 20*20 ms时长的
        self.v_reso = (8.0 / 3) / 50  # 速度分辨率:每次的步长(这里的步长为3)
        self.w_reso = (2 * pi / 3) / 50  # 角速度分辨率:每次的步长(这里的步长为3)
        self.predict_time = 20 * 10  # 预测 (20*10)/20 个 20*20ms [相当于 10个20帧 = 4S]内的 取对应步长的速度 形成的 运动轨迹
        self.goal_factor = 1.0
        self.vel_factor = 1.0
        self.traj_factor = 1.0


# 产生速度空间
def vw_generate(info):
    # generate v,w window for traj prediction
    v_info = [info.v_min, info.v_max,
              info.w_min, info.w_max]

    v_move = [info.x[3] - info.vacc_max * info.dt,
              info.x[3] + info.vacc_max * info.dt,
              info.x[4] - info.wacc_max * info.dt,
              info.x[4] + info.wacc_max * info.dt]

    # 保证速度变化不超过info限制的范围
    vw = [max(v_info[0], v_move[0]), min(v_info[1], v_move[1]),
          max(v_info[2], v_move[2]), min(v_info[3], v_move[3])]

    return vw


# 定义机器人运动模型
# 返回坐标(x,y),偏移角theta,速度v,角速度w
def motion_model(x, u, dt):
    # robot motion model: x,y,theta,v,w
    x[0] += u[0] * dt * cos(x[2])
    x[1] += u[0] * dt * sin(x[2])
    x[2] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


# 依据当前位置及速度，预测轨迹
def traj_calculate(x, u, info):
    traj = np.array(x)
    x_new = np.array(x)  # Caution!!! Don't use like this: xnew = x, it will change x value when run motion_modle below
    time = 0

    while time <= info.predict_time:
        x_new = motion_model(x_new, u, info.dt)
        traj = np.vstack((traj, x_new))
        time += info.dt  # 0.1

    return traj


# 距离目标点评价函数
def goal_evaluate(traj, goal):
    # calculate current pose to goal with euclidean distance
    goal_score = sqrt((traj[-1, 0] - goal[0]) ** 2 + (traj[-1, 1] - goal[1]) ** 2)
    return goal_score


# 速度评价函数
def velocity_evaluate(traj, info):
    # calculate current velocity score
    vel_score = info.v_max - traj[-1, 3]
    return vel_score


# 轨迹距离障碍物的评价函数
def traj_evaluate(traj, obstacles):
    # evaluate current traj with the min distance to obstacles
    min_dis = float("Inf")
    for i in range(len(traj)):
        for ii in range(len(obstacles)):
            current_dist = sqrt((traj[i, 0] - obstacles[ii, 0]) ** 2 + (traj[i, 1] - obstacles[ii, 1]) ** 2)

            if current_dist <= 2 * const.ROBOT_RADIUS_PRODUCT:
                return float("Inf")

            if min_dis >= current_dist:
                min_dis = current_dist

    return 1 / min_dis


def DWA_Core(info: Info):
    x = info.x
    u = np.array([0, 0])
    goal = info.goal

    vw = vw_generate(info)
    obstacles = info.obstacles
    min_score = 10000.0  # 随便设置一下初始的最小评价分数

    # 速度v,w都被限制在速度空间里, 速度从最小开始, 每次取reso步长增量, 直到最大速度为止
    # 每次计算对应的运动轨迹, 分别对这些运动轨迹进行 目标距离, 速度最优, 碰撞距离进行评估
    # 评估分数越低越好
    for v in np.arange(vw[0], vw[1], info.v_reso):
        for w in np.arange(vw[2], vw[3], info.w_reso):
            # calculate traj for each given (v,w)
            traj = traj_calculate(x, [v, w], info)
            # 计算评价函数
            goal_score = info.goal_factor * goal_evaluate(traj, goal)
            vel_score = info.vel_factor * velocity_evaluate(traj, info)
            traj_score = info.traj_factor * traj_evaluate(traj, obstacles)
            # 可行路径不止一条，通过评价函数确定最佳路径
            # 路径总分数 = 距离目标点 + 速度 + 障碍物
            # 分数越低，路径越优
            traj_score = goal_score + vel_score + traj_score
            # evaluate current traj (the score smaller,the traj better)
            if min_score >= traj_score:
                min_score = traj_score
                u = np.array([v, w])

    return u
