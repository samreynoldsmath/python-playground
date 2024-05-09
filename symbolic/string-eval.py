import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def main():
    n = 16
    t = np.linspace(0, 2 * np.pi, 2 * n + 1)
    expr_str = "t*cos(t)"
    vals = get_vals(expr_str, t)
    print(vals)
    plot_vals(t, vals)

def get_vals(expr_str: str, t: np.ndarray) -> np.ndarray:
    t_var = sp.symbols("t")
    expr = sp.sympify(expr_str)
    return np.array([expr.subs(t_var, val) for val in t])

def plot_vals(t: np.ndarray, vals: np.ndarray) -> None:
    plt.plot(t, vals, "k.-")
    plt.show()

if __name__ == "__main__":
    main()