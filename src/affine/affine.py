import numpy as np
import scipy as sp


def is_an_affine_transformation(
    X: np.ndarray, Y: np.ndarray, tol: float = 1e-12
) -> bool:
    """
    Test if there is an affine transformation such that y_j = A x_j + b, where
    x_j is the jth column of X and y_j is the jth column of Y.

    Parameters
    ----------
    X : numpy.ndarray
        The source points, shape (d, m), where d is the dimension of the points
        and m is the number of points.
    Y : numpy.ndarray
        The target points, shape (d, m), where d is the dimension of the points
        and m is the number of points.
    tol : float
        Tolerance against which to test against the Frobenius norm of the
        residule.

    Returns
    -------
    is_affine : bool
        True if and only if y_j = A x_j + b for all j.

    Raises
    ------
    ValueError : When all the points x_j lie on a hyperplane.

    See also
    --------
    get_affine_transformation(X, Y)
    """
    m = X.shape[1]
    A, b = get_affine_transformation(X, Y)
    r = A @ X - Y
    for j in range(m):
        r[:, j] += b
    return np.linalg.norm(r) <= tol * m


def get_affine_transformation(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the affine transformation A x + b that maps x_j to y_j, where x_j is
    the jth column of X and y_j is the jth column of Y.

    This method does not work when all of the points x_j lie on a common
    hyperplane, as there is not a unique affine transformation taking X to Y.
    If this is the case, one can try:
        1. Add another point to X that does not lie in the same hyperplane, and
           add an arbitrary point to Y.
        2. Reduce the dimension of X (and Y).

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

    Raises
    ------
    ValueError : When all the points x_j lie on a hyperplane.

    Notes
    -----
    The affine transformation matrix A and translation vector b are found by
    solving the least squares problem:
        min (1/2) ||A @ X + b - Y||^2
    with the gradients of the objective function given by
        grad_A f = (A X + b ones.T - Y) X.T
        grad_b f = (A X + b ones.T - Y) ones
    Setting the gradients equal to zero and solving leads to the need to invert
        Z = X X.T - (1/m) (X ones) (X ones).T
    which can be seen (via Cauchy-Schwarz) to be singular if and only if
        X.T u = alpha ones
    for some nonzero vector u and scalar alpha, which is equivalent to
    x_j * u = alpha for each j, which is to say that all the x_j's lie in a
    hyperplane.
    """
    # check shapes
    d, m = X.shape
    assert Y.shape == (d, m)
    assert d >= 2
    assert m > d

    # recurring quantities
    ones = np.ones((m,))
    X_ones = X @ ones
    Y_ones = Y @ ones
    YX_T = Y @ X.T

    # invert lease squares matrix
    Z = X @ X.T - np.outer(X_ones, X_ones) / m
    if np.linalg.matrix_rank(Z) < d:
        raise ValueError("Not all the points in X can lie on a hyperplane")

    # proceed with least squares
    U = np.linalg.inv(Z)
    v = -U @ X_ones / m
    w = (1 - np.dot(v, X_ones)) / m

    # solve least squares problem
    A = YX_T @ U + np.outer(Y_ones, v)
    b = YX_T @ v + w * Y_ones

    return A, b


def _test1():
    A_true = np.array([[2, -1], [1, 1]])
    b_true = np.array([1, 1])
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 5]])
    X = np.transpose(X)
    Y = A_true @ X
    for i in range(X.shape[1]):
        Y[:, i] += b_true
    assert is_an_affine_transformation(X, Y)
    print("test 1 passed")


def _test2():
    A_true = np.array([[2, -1], [1, 1]])
    b_true = np.array([1, 1])
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 5]])
    X = np.transpose(X)
    Y = np.array([[1, -2], [2, 3], [3, -4], [4, 5], [5, -5]])
    Y = np.transpose(Y)
    assert not is_an_affine_transformation(X, Y)
    print("test 2 passed")


def _test3():
    A_true = np.array([[2, -1], [1, 1]])
    b_true = np.array([1, 1])
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    X = np.transpose(X)
    Y = A_true @ X
    for i in range(X.shape[1]):
        Y[:, i] += b_true
    try:
        is_an_affine_transformation(X, Y)
    except ValueError as e:
        print("test 3 passed")
        return
    assert False

def _test4():
    t = np.linspace(0, 2 * np.pi)
    m = len(t)
    X = np.zeros((2,m))
    X[0, :] = np.cos(t)
    X[1, :] = np.sin(t)
    Y = np.zeros((2, m))
    Y[0, :] = 2 * X[0, :] - X[1, :] + 1
    Y[1, :] = 3 * X[1,:] - 4
    assert is_an_affine_transformation(X, Y)
    print("test 4 passed")


if __name__ == "__main__":
    print("Running tests for affine.py")
    _test1()
    _test2()
    _test3()
    _test4()
