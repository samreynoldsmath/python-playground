import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
import numpy as np

def main():
    nodes_x, nodes_y = node_coordinates()
    triangulation = tri.Triangulation(nodes_x, nodes_y)
    scalars = function_to_interpolate(nodes_x, nodes_y)
    plot_triangulation(triangulation, scalars)

def node_coordinates(num_boundary_nodes=20, num_interior_nodes=10):
    nodes_x, nodes_y = boundary_nodes(num_boundary_nodes)
    interior_x, interior_y = interior_nodes(num_interior_nodes, nodes_x, nodes_y)
    nodes_x.extend(interior_x)
    nodes_y.extend(interior_y)
    return np.array(nodes_x), np.array(nodes_y)

def boundary_nodes(num_boundary_nodes):
    t = np.linspace(0, 2*np.pi, num_boundary_nodes)
    nodes_x = list(np.cos(t))
    nodes_y = list(np.sin(t))
    return nodes_x, nodes_y

def interior_nodes(num_interior_nodes, nodes_x, nodes_y):
    x_min, x_max, y_min, y_max = bounding_box(nodes_x, nodes_y)
    interior_x = np.linspace(x_min, x_max, num_interior_nodes)
    interior_y = np.linspace(y_min, y_max, num_interior_nodes)
    interior_x, interior_y = np.meshgrid(interior_x, interior_y)
    polygon = np.zeros((len(nodes_x), 2))
    polygon[:, 0] = np.array(nodes_x)
    polygon[:, 1] = np.array(nodes_y)
    is_inside = Path(polygon).contains_points(np.array([interior_x.flatten(), interior_y.flatten()]).T)
    is_inside = is_inside.reshape((num_interior_nodes, num_interior_nodes))
    return interior_x[is_inside], interior_y[is_inside]

def bounding_box(x, y):
    return min(x), max(x), min(y), max(y)

def function_to_interpolate(x, y):
    return (x**2 + y**2) * np.sin(x*y * np.pi)

def plot_triangulation(triangulation, scalars):
    plt.tricontourf(triangulation, scalars)
    plt.triplot(triangulation, '-k')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()