import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

TWO_PI = 2 * np.pi


class Edge:
    t: sp.Expr = sp.symbols("t")
    t_bounds: tuple
    x_expr: sp.Expr
    y_expr: sp.Expr
    x: np.ndarray
    y: np.ndarray

    def __init__(
        self,
        x: str,
        y: str,
        t_bounds: tuple[float, float] = (0.0, 2 * TWO_PI),
    ):
        self.set_parameterization(x, y)
        self.t_bounds = t_bounds

    def set_parameterization(self, x: str, y: str) -> None:
        """
        Define the edge parameterization using strings x,y each being an
        expression in terms of t
        """
        for s in [x, y]:
            self.validate_string_expression(s)
        self.x_expr = sp.parse_expr(x)
        self.y_expr = sp.parse_expr(y)

    def validate_string_expression(self, s: str) -> None:
        """
        Raises a ValueError if the string is not a valid expression in terms of
        t. Specifically, we require the string s to satisfy:
        1. s contains no other variables except t (no "free symbols")
        2. s contains no undefined expressions
        """
        expr = sp.parse_expr(s)
        if expr.free_symbols and self.t not in expr.free_symbols:
            raise ValueError(f"Unable to parse string: '{s}'")

    def sample(self, num_pts):
        t_vals = np.linspace(self.t_bounds[0], self.t_bounds[1], num_pts)
        self.x = self.evaluate_expression(self.x_expr, t_vals)
        self.y = self.evaluate_expression(self.y_expr, t_vals)

    def evaluate_expression(self, expr: sp.Expr, t_vals: np.ndarray) -> np.ndarray:
        vals = np.zeros(t_vals.shape)
        for idx, t_val in enumerate(t_vals):
            vals[idx] = float(expr.subs(self.t, t_val))
        return vals

def example(make_plot: bool=True):
    edge = Edge("t*cos(t)", "t*sin(t)")
    edge.sample(32)
    if make_plot:
        plt.plot(edge.x, edge.y, "k.-")
        plt.show()

if __name__ == "__main__":
    example(make_plot=True)
