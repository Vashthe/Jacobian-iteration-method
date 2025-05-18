import numpy as np


def jacobian_iteration_method(A, b, x0, epsilon=1e-10, max_iter=5000):
    """
    雅可比迭代法求解线性方程组 Ax = b。

    参数:
        A: 系数矩阵 (n x n)
        b: 右侧向量 (1 x n 或 n x 1)
        x0: 初始猜测 (1 x n 或 n x 1)
        epsilon: 收敛阈值 (默认 1e-10)
        max_iter: 最大迭代次数 (默认 5000)
    """
    # 统一输入为行向量
    b = np.atleast_2d(b)
    if b.shape[0] != 1:
        b = b.T
    x_init = np.atleast_2d(x0)
    if x_init.shape[0] != 1:
        x_init = x_init.T

    n = A.shape[0]
    I = np.eye(n)

    # 检查对角元素是否为零
    diag_A = np.diag(A)
    if np.any(diag_A == 0):
        raise ValueError("矩阵 A 的对角元素包含零，雅可比迭代法不适用！")

    # 构造对角矩阵 D 和其逆 D_re
    D = np.diag(diag_A)
    D_re = np.diag(1 / diag_A)

    # 雅可比迭代矩阵和向量
    B = I - D_re @ A
    f = D_re @ b.T  # 注意 b 是行向量，需要转置

    # 迭代过程
    for k in range(1, max_iter + 1):
        x_prev = x_init
        x_init = B @ x_prev.T + f  # x_init 变为列向量
        x_init = x_init.T  # 转回行向量

        # 计算误差
        error = np.max(np.abs(x_init - x_prev))
        if error < epsilon:
            print(f"迭代成功！近似解: {x_init.flatten()}，迭代次数: {k}，最终误差: {error:.2e}")
            return x_init.flatten()

    print(f"迭代失败！未达到精度。最后一次误差: {error:.2e}")
    return x_init.flatten()


# 测试用例
A = np.array([[8, -3, 2],
              [4, 11, -1],
              [6, 3, 12]], dtype=float)
b = np.array([20, 33, 36], dtype=float)  # 允许行向量或列向量
x0 = np.zeros(3)  # 允许行向量或列向量

x_sol = jacobian_iteration_method(A, b, x0)