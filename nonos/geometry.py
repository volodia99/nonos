#!/usr/bin/env python
import numpy as np

DICT_PLANE = {
    "rphi": [(1, 2), "cylindrical"],
    "phir": [(2, 1), "cylindrical"],
    "rz": [(1, 3), "cylindrical"],
    "zr": [(3, 1), "cylindrical"],
    "rtheta": [(1, 2), "spherical"],
    "thetar": [(2, 1), "spherical"],
    "xy": [(1, 2), "cartesian"],
    "xz": [(1, 3), "cartesian"],
    "yz": [(2, 3), "cartesian"],
}


def cart2cyl(X, Y, Z):
    R = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)
    z = Z
    return (R, phi, z)


def cart2sph(X, Y, Z):
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    theta = np.arctan2(np.sqrt(X ** 2 + Y ** 2), Z)
    phi = np.arctan2(Y, X)
    return (r, phi, theta)


def cyl2cart(R, phi, z):
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    Z = z
    return (X, Y, Z)


def cyl2sph(R, phi, z):
    r = np.sqrt(R ** 2 + z ** 2)
    theta = np.arctan2(R, z)
    phi = phi
    return (r, phi, theta)


def sph2cart(r, theta, phi):
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)
    return (X, Z, Y)


def sph2cyl(r, theta, phi):
    R = r * np.sin(theta)
    phi = phi
    z = r * np.cos(theta)
    return (R, z, phi)


def noproj(x, y, z):
    return [x, y, z]


def meshgridFromPlane(coord, k, l, DEFAULT):
    lgrid, kgrid = np.meshgrid(coord[l - 1], coord[k - 1])
    tot = {1, 2, 3}
    m = list(tot ^ {k, l})[0]
    mgrid = DEFAULT[m - 1]
    grid = [kgrid, lgrid, mgrid]
    return grid
