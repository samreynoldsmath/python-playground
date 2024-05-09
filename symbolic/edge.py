import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

t = sp.Symbol("t")


class Edge:
    t_bounds: tuple
    x_expr: sp.Expr
    y_expr: sp.Expr
    x: np.ndarray
    y: np.ndarray

    def __init__(
        self,
        x_expr: sp.Expr,
        y_expr: sp.Expr,
        t_bounds: tuple[float, float] = (0.0, 2 * np.pi),
    ):
        self.x_expr = x_expr
        self.y_expr = y_expr
        self.t_bounds = t_bounds

    def sample(self, num_pts):
        t_vals = np.linspace(self.t_bounds[0], self.t_bounds[1], num_pts)
        self.x = np.array([self.x_expr.subs(t, val) for val in t_vals])
        self.y = np.array([self.y_expr.subs(t, val) for val in t_vals])


if __name__ == "__main__":
    x_expr = t * sp.cos(t)
    y_expr = t * sp.sin(t)
    edge = Edge(x_expr, y_expr)
    edge.sample(100)
    plt.plot(edge.x, edge.y)
    plt.show()
