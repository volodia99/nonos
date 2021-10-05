import numpy as np


def cartesian_to_cylindrical(X, Y, Z):
    R = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)
    return R, phi, Z


def cartesian_to_spherical(X, Y, Z):
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    theta = np.arctan2(np.sqrt(X ** 2 + Y ** 2), Z)
    phi = np.arctan2(Y, X)
    return (r, phi, theta)


def cylindrical_to_cartesian(R, phi, z):
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return X, Y, z


def cylindrical_to_spherical(R, phi, z):
    r = np.sqrt(R ** 2 + z ** 2)
    theta = np.arctan2(R, z)
    return (r, phi, theta)


def spherical_to_cartesian(r, theta, phi):
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)
    return X, Z, Y


def spherical_to_cylindrical(r, theta, phi):
    R = r * np.sin(theta)
    z = r * np.cos(theta)
    return R, z, phi


def meshgrid_from_plane(coord, k, l, default):
    lgrid, kgrid = np.meshgrid(coord[l - 1], coord[k - 1])
    tot = {1, 2, 3}
    m = list(tot ^ {k, l})[0]
    mgrid = default[m - 1]
    return kgrid, lgrid, mgrid


def get_keys_from_geomtransforms(dictionary, values):
    """
    you give this function a dictionary
    of the type GEOM_TRANSFORMS[structure]
    (see dictionary just under the function)
    + a list of values (ex: plane=(1,2,3) and geometry="cylindrical"),
    and it finds the corresponding key (here in the example : "rphi").
    It is used to give the good name for the savefig,
    knowing the plane and the geometry.
    """
    list_items = dictionary.items()
    for item in list_items:
        if item[1][:-1] == values:
            itemf = item[0]
    return itemf


GEOM_TRANSFORMS = {
    "polar": {
        "rphi": [(1, 2, 3), "cylindrical", None],
        "rz": [(1, 3, 2), "cylindrical", None],
        "rtheta": [(1, 3, 2), "spherical", cylindrical_to_spherical],
        "xy": [(1, 2, 3), "cartesian", cylindrical_to_cartesian],
        "xz": [(1, 3, 2), "cartesian", cylindrical_to_cartesian],
        "yz": [(2, 3, 1), "cartesian", cylindrical_to_cartesian],
    },
    "spherical": {
        "rtheta": [(1, 2, 3), "spherical", None],
        "rphi": [(1, 3, 2), "spherical", None],
        "rz": [(1, 2, 3), "cylindrical", spherical_to_cylindrical],
        "xy": [(1, 3, 2), "cartesian", spherical_to_cartesian],
        "xz": [(1, 2, 3), "cartesian", spherical_to_cartesian],
        "yz": [(2, 3, 1), "cartesian", spherical_to_cartesian],
    },
    "cartesian": {
        "xy": [(1, 2, 3), "cartesian", None],
        "xz": [(1, 3, 2), "cartesian", None],
        "yz": [(2, 3, 1), "cartesian", None],
        "rphi": [(1, 2, 3), "cylindrical", cartesian_to_cylindrical],
        "rz": [(1, 3, 2), "cylindrical", cartesian_to_cylindrical],
        "rtheta": [(1, 3, 2), "spherical", cartesian_to_spherical],
    },
}
