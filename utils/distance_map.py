import numpy as np


def euclidean_distance(p1, p2, scale=4.5):
    distance = 0
    for dim in range(len(p1)):
        distance += (p1[dim] - p2[dim]) ** 2
    return np.sqrt(distance) * scale


def create_distance_map(shape, points, omega=255):
    dm = np.full(shape, omega)

    for p in points:
        x_min = 0
        x_max = shape[0]
        y_min = 0
        y_max = shape[1]

        for x in range(p[0], 0, -1):
            if euclidean_distance((x, p[1]), (p[0], p[1])) > omega:
                x_min = x
                break
        for x in range(p[0], shape[0]):
            if euclidean_distance((x, p[1]), (p[0], p[1])) > omega:
                x_max = x
                break

        for y in range(p[1], 0, -1):
            if euclidean_distance((p[0], y), (p[0], p[1])) > omega:
                y_min = y
                break
        for y in range(p[1], shape[1]):
            if euclidean_distance((p[0], y), (p[0], p[1])) > omega:
                y_max = y
                break

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                dm[x, y] = min(dm[x, y], euclidean_distance((x, y), (p[0], p[1])))

    return np.abs(np.array(dm) - 255.0)
