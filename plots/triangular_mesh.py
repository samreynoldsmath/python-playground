import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
import numpy as np


def main():
    num_edge_nodes = 128
    int_mesh_size = (64, 64)

    # make some example coordinates
    boundary_x, boundary_y = boundary_nodes(num_edge_nodes)

    # partition the boundary nodes into the exterior and two holes
    outer_x = boundary_x[:num_edge_nodes]
    outer_y = boundary_y[:num_edge_nodes]

    hole1_x = boundary_x[num_edge_nodes : 2 * num_edge_nodes]
    hole1_y = boundary_y[num_edge_nodes : 2 * num_edge_nodes]

    hole2_x = boundary_x[2 * num_edge_nodes :]
    hole2_y = boundary_y[2 * num_edge_nodes :]

    # find the interior nodes
    interior_x, interior_y = interior_nodes(
        outer_x,
        outer_y,
        [hole1_x, hole2_x],
        [hole1_y, hole2_y],
        int_mesh_size,
        radius=-1e-1,
    )

    # combine the boundary and interior nodes
    nodes_x, nodes_y = node_coordinates(
        boundary_x, boundary_y, interior_x, interior_y
    )

    # create a triangulation
    triangulation = tri.Triangulation(nodes_x, nodes_y)
    triangulation = remove_holes(
        triangulation, outer_x, outer_y, [hole1_x, hole2_x], [hole1_y, hole2_y]
    )

    # define a function to interpolate
    vals = function_to_interpolate(nodes_x, nodes_y)

    # plot the nodes by type
    plot_nodes(nodes_x, nodes_y, interior_x, interior_y, num_edge_nodes)

    # plot the triangulation
    plot_triangulation(
        triangulation,
        vals,
        outer_x,
        outer_y,
        [hole1_x, hole2_x],
        [hole1_y, hole2_y],
    )


def remove_holes(triangulation, outer_x, outer_y, holes_x, holes_y):
    """
    The midpoint of each edge is tested to see if it lies inside the domain.
    If it lies outside the domain, the edge is removed.
    """
    mask = np.zeros(triangulation.triangles.shape[0], dtype=bool)
    for t in range(triangulation.triangles.shape[0]):
        for i in range(3):
            if mask[t]:
                continue
            a = triangulation.triangles[t, i]
            b = triangulation.triangles[t, (i + 1) % 3]
            mid_x = 0.5 * (triangulation.x[a] + triangulation.x[b])
            mid_y = 0.5 * (triangulation.y[a] + triangulation.y[b])
            mask[t] = not point_is_inside(
                mid_x, mid_y, outer_x, outer_y, holes_x, holes_y, 1e-8
            )
    triangulation.set_mask(mask)
    return triangulation


def remove_holes_bad(triangulation, hole_indices):
    num_triangles = triangulation.triangles.shape[0]
    mask = np.zeros(num_triangles, dtype=bool)
    for t in range(num_triangles):
        if np.all(np.in1d(triangulation.triangles[t], hole_indices)):
            mask[t] = True
    triangulation.set_mask(mask)
    return triangulation


def boundary_nodes(num_edge_nodes):
    t = np.linspace(0, 2 * np.pi, num_edge_nodes)
    # exterior boundary
    nodes_x = list(np.cos(t))
    nodes_y = list(np.sin(t))
    # hole boundary
    nodes_x.extend(list(0.3 * np.cos(-t) + 0.5))
    nodes_y.extend(list(0.3 * np.sin(-t) + 0.2))
    # hole boundary
    r = 0.3 + 0.25 * np.sin(2 * t)
    nodes_x.extend(list(r * np.cos(-t) - 0.4))
    nodes_y.extend(list(r * np.sin(-t) - 0.2))
    return nodes_x, nodes_y


def node_coordinates(boundary_x, boundary_y, interior_x, interior_y):
    nodes_x = list(boundary_x)
    nodes_y = list(boundary_y)
    nodes_x.extend(interior_x)
    nodes_y.extend(interior_y)
    return np.array(nodes_x), np.array(nodes_y)


def interior_nodes(
    outer_x, outer_y, holes_x, holes_y, int_mesh_size=(10, 10), radius=1e-1
):
    # set up a grid of points in the bounding box
    x_min, x_max, y_min, y_max = bounding_box(outer_x, outer_y)
    interior_x = np.linspace(x_min, x_max, int_mesh_size[0])
    interior_y = np.linspace(y_min, y_max, int_mesh_size[1])
    interior_x, interior_y = np.meshgrid(interior_x, interior_y)

    # find points inside the outer boundary
    is_inside = np.zeros(int_mesh_size, dtype=bool)
    for i in range(int_mesh_size[0]):
        for j in range(int_mesh_size[1]):
            is_inside[i, j] = point_is_inside(
                interior_x[i, j],
                interior_y[i, j],
                outer_x,
                outer_y,
                holes_x,
                holes_y,
                radius,
            )

    # return the interior points
    is_inside = is_inside.reshape(interior_x.shape)
    return interior_x[is_inside], interior_y[is_inside]


def point_is_inside_simple(point_x, point_y, path_x, path_y, radius):
    polygon = np.zeros((len(path_x), 2))
    polygon[:, 0] = np.array(path_x)
    polygon[:, 1] = np.array(path_y)
    return Path(polygon).contains_point((point_x, point_y), radius=radius)


def point_is_inside(
    point_x, point_y, outer_x, outer_y, holes_x, holes_y, radius
):
    if not point_is_inside_simple(point_x, point_y, outer_x, outer_y, radius):
        return False
    for hole_x, hole_y in zip(holes_x, holes_y):
        if point_is_inside_simple(point_x, point_y, hole_x, hole_y, radius):
            return False
    return True


def bounding_box(x, y):
    return min(x), max(x), min(y), max(y)


def function_to_interpolate(x, y):
    return (x**2 + y**2) * np.sin(x * y * np.pi)


def plot_triangulation(
    triangulation, scalars, outer_x, outer_y, holes_x=None, holes_y=None
):
    plt.tricontourf(triangulation, scalars)
    plt.triplot(triangulation, "-k")
    plt.plot(outer_x, outer_y, "b--")
    if holes_x is not None:
        for hole_x, hole_y in zip(holes_x, holes_y):
            plt.plot(hole_x, hole_y, "b--")
    plt.colorbar()
    plt.axis("equal")
    plt.show()


def plot_nodes(nodes_x, nodes_y, interior_x, interior_y, num_edge_nodes):
    plt.figure()
    for k in range(3):
        plt.plot(
            nodes_x[k * num_edge_nodes : (k + 1) * num_edge_nodes],
            nodes_y[k * num_edge_nodes : (k + 1) * num_edge_nodes],
            "b-",
        )
    plt.scatter(interior_x, interior_y, c="r")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
