import numpy as np


def cartesian_to_cylindrical(X, Y, Z):
    R = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)
    z = Z
    return (R, phi, z)


def cartesian_to_spherical(X, Y, Z):
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    theta = np.arctan2(np.sqrt(X ** 2 + Y ** 2), Z)
    phi = np.arctan2(Y, X)
    return (r, phi, theta)


def cylindrical_to_cartesian(R, phi, z):
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    Z = z
    return (X, Y, Z)


def cylindrical_to_spherical(R, phi, z):
    r = np.sqrt(R ** 2 + z ** 2)
    theta = np.arctan2(R, z)
    phi = phi
    return (r, phi, theta)


def spherical_to_cartesian(r, theta, phi):
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)
    return (X, Z, Y)


def spherical_to_cylindrical(r, theta, phi):
    R = r * np.sin(theta)
    phi = phi
    z = r * np.cos(theta)
    return (R, z, phi)


def no_op(*args):
    return args


def meshgridFromPlane(coord, k, l, DEFAULT):
    lgrid, kgrid = np.meshgrid(coord[l - 1], coord[k - 1])
    tot = {1, 2, 3}
    m = list(tot ^ {k, l})[0]
    mgrid = DEFAULT[m - 1]
    return [kgrid, lgrid, mgrid]


DICT_PLANE = {
    "cylindrical": {
        "rphi": [(1, 2), "cylindrical", no_op],
        "rz": [(1, 3), "cylindrical", no_op],
        "rtheta": [(1, 3), "spherical", cylindrical_to_spherical],
        "xy": [(1, 2), "cartesian", cylindrical_to_cartesian],
        "xz": [(1, 3), "cartesian", cylindrical_to_cartesian],
        "yz": [(2, 3), "cartesian", cylindrical_to_cartesian],
    },
    "spherical": {
        "rtheta": [(1, 2), "spherical", no_op],
        "rphi": [(1, 3), "spherical", no_op],
        "rz": [(1, 2), "cylindrical", spherical_to_cylindrical],
        "xy": [(1, 3), "cartesian", spherical_to_cartesian],
        "xz": [(1, 2), "cartesian", spherical_to_cartesian],
        "yz": [(2, 3), "cartesian", spherical_to_cartesian],
    },
    "cartesian": {
        "xy": [(1, 2), "cartesian", no_op],
        "xz": [(1, 3), "cartesian", no_op],
        "yz": [(2, 3), "cartesian", no_op],
        "rphi": [(1, 2), "cylindrical", cartesian_to_cylindrical],
        "rz": [(1, 3), "cylindrical", cartesian_to_cylindrical],
        "rtheta": [(1, 3), "spherical", cartesian_to_spherical],
    },
}
