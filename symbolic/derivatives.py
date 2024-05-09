import sympy as sp

def main():
    t = sp.symbols("t")
    expr = t * sp.cos(t)

    # first derivative
    d_expr = sp.diff(expr, t)
    print(d_expr)

    # second derivative
    dd_expr = sp.diff(d_expr, t)
    print(dd_expr)

if __name__ == "__main__":
    main()