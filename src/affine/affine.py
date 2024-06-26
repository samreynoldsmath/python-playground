import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def get_affine_transformation(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the affine transformation matrix that maps X to Y via least squares.

    Parameters
    ----------
    X : numpy.ndarray
        The source points, shape (d, m), where d is the dimension of the points
        and m is the number of points.
    Y : numpy.ndarray
        The target points, shape (d, m), where d is the dimension of the points
        and m is the number of points.

    Returns
    -------
    A : numpy.ndarray
        The affine transformation matrix, shape (d, d).
    b : numpy.ndarray
        The translation vector, shape (d,).

    Notes
    -----
    The affine transformation matrix A and translation vector b are found by
    solving the least squares problem:
        min ||A @ X + b - Y||^2
    """
    # get shape
    d, m = X.shape
    assert Y.shape == (d, m)
    assert d >= 2
    assert m > d

    # check that X is full rank
    r = np.linalg.matrix_rank(X)
    assert r == d

    # recurring quantities
    ones = np.ones((m,))
    X_ones = X @ ones
    Y_ones = Y @ ones
    X_T = np.transpose(X)
    YX_T = Y @ X_T

    # invert lease squares matrix
    Z = X @ X_T - np.outer(X_ones, X_ones) / m

    # check that Z is invertible
    if np.linalg.matrix_rank(Z) < d:
        # this happens if and only if the x points lie in a hyperplane
        return get_affine_transformation_with_points_in_hyperplane(X, Y)

    # proceed with least squares
    U = np.linalg.inv(Z)
    v = - U @ X_ones / m
    w = (1 - np.dot(v, X_ones)) / m

    # solve least squares problem
    A = YX_T @ U + np.outer(Y_ones, v)
    b = YX_T @ v + w * Y_ones

    return A, b


def get_affine_transformation_with_points_in_hyperplane(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, alpha = get_hyperplane(X)
    d, beta = get_hyperplane(Y)
    A = np.outer(c, d)
    b = (beta - alpha) * d
    return A, b


def get_hyperplane(X: np.ndarray) -> tuple[np.ndarray, float]:
    # TODO: implement


if __name__ == "__main__":
    print("Running tests for affine.py")

    # test
    A_true = np.array([[2, 2], [1, 1]])
    b_true = np.array([1, 1])

    X = np.array([[0,0], [1, 0], [0, 1], [1, 1]])
    X = np.transpose(X)
    Y = A_true @ X
    for i in range(X.shape[1]):
        Y[:, i] += b_true

    print(f"X = \n{X}")
    print(f"Y = \n{Y}")

    A, b = get_affine_transformation(X, Y)

    print(f"A = \n{A}")
    print(f"b = \n{b}")

    Y_approx = A @ X
    for i in range(X.shape[1]):
        Y_approx[:, i] += b

    print(f"Y_approx = \n{Y_approx}")
    print(f"Y_error = \n{Y - Y_approx}")
