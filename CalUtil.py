import numpy as np


def update_future_value(workbenches: [], waiting_benches: []):
    for wb in waiting_benches:
        wb.update_future_value(workbenches)


class BallTreeNode:
    def __init__(self, points):
        self.points = points
        self.n = points.shape[0]
        self.radius = 0
        self.center = np.zeros(points.shape[1])
        self.left = None
        self.right = None
        self.build()

    def build(self):
        self.center = np.mean(self.points, axis=0)
        self.radius = np.max(np.linalg.norm(self.points - self.center, axis=1))
        if self.n > 1:
            # Split the points into two subsets
            left_indices = np.random.choice(self.n, self.n // 2, replace=False)
            right_indices = np.setdiff1d(np.arange(self.n), left_indices)
            left_points = self.points[left_indices]
            right_points = self.points[right_indices]

            # Build left and right subtrees
            self.left = BallTreeNode(left_points)
            self.right = BallTreeNode(right_points)

    def search(self, query_point, k=1):
        dist_to_center = np.linalg.norm(query_point - self.center)
        if dist_to_center > self.radius:
            # Query point is outside of this node's bounding sphere
            return [], []
        elif self.left is None and self.right is None:
            # Leaf node, return all points in the node
            return self.points, np.linalg.norm(self.points - query_point, axis=1)
        elif self.left is None:
            # Query right subtree
            right_points, right_dists = self.right.search(query_point, k)
            return right_points, right_dists
        elif self.right is None:
            # Query left subtree
            left_points, left_dists = self.left.search(query_point, k)
            return left_points, left_dists
        else:
            # Query both subtrees
            left_points, left_dists = self.left.search(query_point, k)
            right_points, right_dists = self.right.search(query_point, k)

            # Combine results from left and right subtrees
            combined_points = np.vstack((left_points, right_points))
            combined_dists = np.hstack((left_dists, right_dists))

            # Find the k closest points among the combined results
            indices = np.argsort(combined_dists)[:k]
            closest_points = combined_points[indices]
            closest_dists = combined_dists[indices]

            return closest_points, closest_dists


class BallTree:
    def __init__(self, points):
        self.root = BallTreeNode(points)

    def query(self, query_point, k=1):
        return self.root.search(query_point, k)


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

    we use ball-tree to solve the nearest node finding process
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