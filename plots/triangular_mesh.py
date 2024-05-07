import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
import numpy as np

def main():
    # make some example coordinates
    num_edge_nodes = 20
    boundary_x, boundary_y = boundary_nodes(num_edge_nodes)

    # partition the boundary nodes into the exterior and two holes
    outer_x, outer_y = boundary_x[:num_edge_nodes], boundary_y[:num_edge_nodes]
    hole1_x, hole1_y = boundary_x[num_edge_nodes:2*num_edge_nodes], boundary_y[num_edge_nodes:2*num_edge_nodes]
    hole2_x, hole2_y = boundary_x[2*num_edge_nodes:], boundary_y[2*num_edge_nodes:]
    hole_indices = range(num_edge_nodes, 3*num_edge_nodes)

    # find the interior nodes
    interior_x, interior_y = interior_nodes(outer_x, outer_y, [hole1_x, hole2_x], [hole1_y, hole2_y], int_mesh_size=(10, 10), radius=-1e-3)

    # combine the boundary and interior nodes
    nodes_x, nodes_y = node_coordinates(boundary_x, boundary_y, interior_x, interior_y)

    # plot the nodes by type
    plt.figure()
    for k in range(3):
        plt.plot(nodes_x[k*num_edge_nodes:(k+1)*num_edge_nodes], nodes_y[k*num_edge_nodes:(k+1)*num_edge_nodes], 'b-')
    plt.scatter(interior_x, interior_y, c='r')
    plt.show()

    # create a triangulation
    triangulation = tri.Triangulation(nodes_x, nodes_y)
    triangulation = remove_holes(triangulation, hole_indices)

    # define a function to interpolate
    scalars = function_to_interpolate(nodes_x, nodes_y)

    # plot the triangulation
    plot_triangulation(triangulation, scalars)

def remove_holes(triangulation, hole_indices):
    num_triangles = triangulation.triangles.shape[0]
    mask = np.zeros(num_triangles, dtype=bool)
    for t in range(num_triangles):
        if np.all(np.in1d(triangulation.triangles[t], hole_indices)):
            mask[t] = True
    triangulation.set_mask(mask)
    return triangulation


def boundary_nodes(num_edge_nodes):
    t = np.linspace(0, 2*np.pi, num_edge_nodes)
    # exterior boundary
    nodes_x = list(np.cos(t))
    nodes_y = list(np.sin(t))
    # hole boundary
    nodes_x.extend(list(0.3*np.cos(-t) + 0.4))
    nodes_y.extend(list(0.3*np.sin(-t)))
    # hole boundary
    nodes_x.extend(list(0.2*np.cos(-t) - 0.4))
    nodes_y.extend(list(0.2*np.sin(-t) - 0.2))
    return nodes_x, nodes_y

def node_coordinates(boundary_x, boundary_y, interior_x, interior_y):
    nodes_x = list(boundary_x)
    nodes_y = list(boundary_y)
    nodes_x.extend(interior_x)
    nodes_y.extend(interior_y)
    return np.array(nodes_x), np.array(nodes_y)

def interior_nodes(outer_x, outer_y, holes_x, holes_y, int_mesh_size=(10, 10),radius=1e-1):
    # set up a grid of points in the bounding box
    x_min, x_max, y_min, y_max = bounding_box(outer_x, outer_y)
    interior_x = np.linspace(x_min, x_max, int_mesh_size[0])
    interior_y = np.linspace(y_min, y_max, int_mesh_size[1])
    interior_x, interior_y = np.meshgrid(interior_x, interior_y)

    # find points inside the outer boundary
    is_inside = are_points_inside(outer_x, outer_y, interior_x.flatten(), interior_y.flatten(), radius)

    # remove points inside the holes
    for hole_x, hole_y in zip(holes_x, holes_y):
        is_in_hole = are_points_inside(hole_x, hole_y, interior_x.flatten(), interior_y.flatten(), radius)
        is_inside = np.logical_and(is_inside, np.logical_not(is_in_hole))

    is_inside = is_inside.reshape(interior_x.shape)
    return interior_x[is_inside], interior_y[is_inside]

def are_points_inside(path_x, path_y, points_x, points_y, radius):
    polygon = np.zeros((len(path_x), 2))
    polygon[:, 0] = np.array(path_x)
    polygon[:, 1] = np.array(path_y)
    return Path(polygon).contains_points(np.array([points_x, points_y]).T, radius=radius)

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