import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.path import Path
import numpy as np


def main():
    num_edge_nodes = 32
    int_mesh_size = (16, 16)
    radius = 0.1

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
    interior_x_raw, interior_y_raw, is_inside = interior_nodes_raw(
        outer_x,
        outer_y,
        [hole1_x, hole2_x],
        [hole1_y, hole2_y],
        int_mesh_size,
        radius,
    )

    # remove points too close to the boundary
    interior_x, interior_y = interior_nodes_clean(
        interior_x_raw, interior_y_raw, is_inside
    )

    # combine the boundary and interior nodes
    nodes_x, nodes_y = node_coordinates(boundary_x, boundary_y, interior_x, interior_y)

    # create a triangulation
    triangulation = tri.Triangulation(nodes_x, nodes_y)
    triangulation = remove_holes(
        triangulation, outer_x, outer_y, [hole1_x, hole2_x], [hole1_y, hole2_y]
    )

    # define a function to interpolate
    vals = function_to_interpolate(nodes_x, nodes_y)
    interior_vals_raw = function_to_interpolate(interior_x_raw, interior_y_raw)
    interior_vals_raw[np.logical_not(is_inside)] = np.nan

    # plot the nodes by type
    plt.figure()
    plot_nodes(
        interior_x, interior_y, outer_x, outer_y, [hole1_x, hole2_x], [hole1_y, hole2_y]
    )

    # plot the function only on the interior nodes
    plt.figure()
    plot_classic(
        interior_vals_raw,
        interior_x_raw,
        interior_y_raw,
        outer_x,
        outer_y,
        [hole1_x, hole2_x],
        [hole1_y, hole2_y],
    )

    # plot the triangulation
    plt.figure()
    plot_triangulation(
        triangulation,
        outer_x,
        outer_y,
        [hole1_x, hole2_x],
        [hole1_y, hole2_y],
        interior_x,
        interior_y,
    )

    # plot the interpolated function
    plt.figure()
    plot_interpolated(
        triangulation, vals, outer_x, outer_y, [hole1_x, hole2_x], [hole1_y, hole2_y]
    )

    # show the plots
    plt.show()


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


def boundary_nodes(num_edge_nodes):
    t = np.linspace(0, 2 * np.pi, num_edge_nodes)
    # exterior boundary
    nodes_x = list(np.cos(t))
    nodes_y = list(np.sin(t))
    # hole boundary
    nodes_x.extend(list(0.3 * np.cos(1 - t) + 0.5))
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


def interior_nodes_raw(outer_x, outer_y, holes_x, holes_y, int_mesh_size, radius):
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

    # eliminate points too close to the boundary
    boundary_x = np.concatenate((outer_x, *holes_x))
    boundary_y = np.concatenate((outer_y, *holes_y))
    for i in range(int_mesh_size[0]):
        for j in range(int_mesh_size[1]):
            if not is_inside[i, j]:
                continue
            for x, y in zip(boundary_x, boundary_y):
                dist = np.sqrt(
                    (interior_x[i, j] - x) ** 2 + (interior_y[i, j] - y) ** 2
                )
                if dist < np.abs(radius):
                    is_inside[i, j] = False
                    break

    # return the interior points
    is_inside = is_inside.reshape(interior_x.shape)
    return interior_x, interior_y, is_inside


def interior_nodes_clean(interior_x, interior_y, is_inside):
    return interior_x[is_inside], interior_y[is_inside]


def point_is_inside_simple(point_x, point_y, path_x, path_y, radius):
    polygon = np.zeros((len(path_x), 2))
    polygon[:, 0] = np.array(path_x)
    polygon[:, 1] = np.array(path_y)
    return Path(polygon).contains_point((point_x, point_y), radius=radius)


def point_is_inside(point_x, point_y, outer_x, outer_y, holes_x, holes_y, radius):
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


def plot_interpolated(triangulation, scalars, outer_x, outer_y, holes_x, holes_y):
    plt.tricontourf(triangulation, scalars)
    plt.plot(outer_x, outer_y, "k-")
    for hole_x, hole_y in zip(holes_x, holes_y):
        plt.plot(hole_x, hole_y, "k-")
    plt.axis("equal")


def plot_triangulation(
    triangulation, outer_x, outer_y, holes_x, holes_y, interior_x, interior_y
):
    plt.triplot(triangulation, "-k")
    plot_nodes(
        interior_x,
        interior_y,
        outer_x,
        outer_y,
        holes_x,
        holes_y,
    )
    plt.axis("equal")


def plot_nodes(interior_x, interior_y, outer_x, outer_y, holes_x, holes_y):
    plt.scatter(interior_x, interior_y, c="r")
    plt.plot(outer_x, outer_y, "bo")
    for hole_x, hole_y in zip(holes_x, holes_y):
        plt.plot(hole_x, hole_y, "bo")
    plt.axis("equal")


def plot_classic(
    interior_vals, interior_x, interior_y, outer_x, outer_y, holes_x, holes_y
):
    plt.contourf(interior_x, interior_y, interior_vals)
    plt.plot(outer_x, outer_y, "k-")
    for hole_x, hole_y in zip(holes_x, holes_y):
        plt.plot(hole_x, hole_y, "k-")
    plt.axis("equal")


if __name__ == "__main__":
    main()
