import numpy as np
import torch
from operator import itemgetter

class Theta_Lie_Qun:

    def __init__(self, theta_1, theta_2) -> None:
        self.theta = np.deg2rad(0)
        self.theta_1 = np.deg2rad(theta_1)
        self.theta_2 = np.deg2rad(theta_2)
        self.axis = np.array([0, 0, 1])

    @property
    def R(self):
        return self.convert_theta_r(self.axis, self.theta)

    @property
    def R1(self):
        return self.convert_theta_r(self.axis, self.theta_1)

    @property
    def R2(self):
        return self.convert_theta_r(self.axis, self.theta_2)

    @property
    def omega(self):
        tr = np.trace(self.R1 @ self.R2.T) - 1
        cos_theta = tr / 2
        sin_theta = np.sqrt(1 - cos_theta**2)
        return (1 / (2 * sin_theta)) * (self.R2.T @ self.R1 - self.R1.T @ self.R2)

    @property
    def diff1(self):
        return self.R @ self.R1.T
        tr = np.trace(self.R @ self.R1.T) - 1
        cos_theta = tr / 2
        sin_theta = np.sqrt(1 - cos_theta**2)
        return (1 / (2 * sin_theta)) * (self.R1.T @ self.R - self.R.T @ self.R1)

    @property
    def diff2(self):
        return self.R @ self.R2.T
        tr = np.trace(self.R @ self.R2.T) - 1
        cos_theta = tr / 2
        sin_theta = np.sqrt(1 - cos_theta**2)
        return (1 / (2 * sin_theta)) * (self.R1.T @ self.R - self.R.T @ self.R1)

    @staticmethod
    def convert_theta_r(axis: np.ndarray, theta):
        """
        绕任意轴旋转任意角度的旋转矩阵
        :param axis: 旋转轴的方向，为一个三维向量
        :param theta: 旋转角度，单位为弧度
        :return: 旋转矩阵
        """
        if not isinstance(axis, np.ndarray):
            axis = np.array(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))  # 将旋转轴单位化

        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)

        # 计算旋转矩阵
        return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                         [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

    @staticmethod
    def convert_theta_r2(axis: np.ndarray, theta):
        """
        绕任意轴旋转任意角度的旋转矩阵
        :param axis: 旋转轴的方向，为一个三维向量
        :param theta: 旋转角度，单位为弧度
        :return: 旋转矩阵
        """
        if not isinstance(axis, np.ndarray):
            axis = np.array(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))  # 将旋转轴单位化

        nx, ny, nz = axis
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([[(1 - cos_theta) * nx**2 + cos_theta, (1 - cos_theta) * nx * ny - nz * sin_theta,
                          (1 - cos_theta) * nx * nz + ny * sin_theta],
                         [(1 - cos_theta) * nx * ny + nz * sin_theta, (1 - cos_theta) * ny**2 + cos_theta,
                          (1 - cos_theta) * ny * nz - nx * sin_theta],
                         [(1 - cos_theta) * nx * nz - ny * sin_theta, (1 - cos_theta) * ny * nz + nx * sin_theta,
                          (1 - cos_theta) * nz**2 + cos_theta]])

    @staticmethod
    def matrix_to_vector(A):
        """
        将 3x3 的反对称矩阵转化为一个 3 维向量
        :param A: 3x3 反对称矩阵
        :return: 3 维向量
        """
        v = np.array([A[2, 1], A[0, 2], A[1, 0]])
        return v


def cal_distance(theta1, theta2):
    tan1 = torch.tensor(np.deg2rad(theta1)).tan()
    tan2 = torch.tensor(np.deg2rad(theta2)).tan()
    return (tan2 - tan1)**2 / (1 + tan2**2 + tan1**2)

MAP = {
    'front': 'CAM_PBQ_FRONT_FISHEYE',
    'left': 'CAM_PBQ_LEFT_FISHEYE',
    'rear': 'CAM_PBQ_REAR_FISHEYE',
    'right': 'CAM_PBQ_REAR_RIGHT',
    'extrinsics': 'camera_to_vehicle_extrinsics',
    'intrinsics': 'intrinsics'
}

if __name__ == "__main__":
    # theta_lie_qun = Theta_Lie_Qun(320, 1)
    # print(theta_lie_qun.diff1,"\n", theta_lie_qun.diff2)
    # print(theta_lie_qun.diff1, theta_lie_qun.diff2)
    # print(cal_distance(90, 89))
    # print(cal_distance(90, -89))
    # print(cal_distance(90, 1))
    # print(cal_distance(90, 45))
    getter = itemgetter('front', 'left')
    print(getter(MAP))