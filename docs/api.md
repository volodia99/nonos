# Programmatic usage (API)

The core class of nonos is `GasDataSet` that you need to import from `nonos.api`, which takes the form:

```python
ds = GasDataset(
    on:int,
    *,
    directory:str,
    geometry:str,
    code:str,
    inifile:str,
)
```

* `on`: positional argument that corresponds to the output number associated with the VTK file `f"data.{on:04d}.vtk"`
* `directory`: keyword-only argument that corresponds to the working directory where the output file is (default: `""`).
* `geometry`: keyword-only argument if the geometry is not recognized.
* `code` and `inifile`: keyword-only arguments to give if the parameter file is not recognized (i.e. different from idefix.ini for idefix, variables.par for fargo3d and pluto.ini for pluto). `code` can be `"idefix"`, `"fargo3d"`, `"fargo-adsg"` or `"pluto"`.

!!! example "Examples"

    * For a simulation with fargo-adsg and a parameter file "template.par" and for the output number 0, you need to do:
    ```python
    ds = GasDataset(
        0,
        code="fargo-adsg",
        inifile="template.par",
    )
    ```

    * For a simulation performed with idefix in the directory "path/to/output", a parameter file "idefix-rkl.ini" and for the output number 10, you need to do:
    ```python
    ds = GasDataset(
        10,
        directory="path/to/output",
        code="idefix",
        inifile="idefix-rkl.ini",
    )
    ```

## Full examples

!!! example "Example 1"

    After going to the nonos directory, and opening ipython, we import GasDataSet.
    ```python
    import matplotlib.pyplot as plt
    from nonos.api import GasDataSet
    ```

    We use the class GasDataSet which takes as argument the output number of the output file given by idefix/pluto/fargo.
    ```python
    ds = GasDataSet(43, geometry="polar", directory="tests/data/idefix_planet3d")
    ```

    `ds` contains in particular a dictionary with the different fields. You can check what fields are included in `ds` by running `ds.keys()`. Let's say you want to perform a vertical slice of the density in the midplane, plot the result in the `xy` plane and rotate the grid given the planet number 0 (which orbit is described in the planet0.dat file):
    ```python
    dsvm = ds["RHO"].vertical_at_midplane().map("x", "y", planet_corotation=0)
    ```

    dsop is now a Plotable object. We can e.g. represent its log10, with a given colormap, and display the colorbar by adding the argument `title`.
    ```python
    plt.close("all")
    fig, ax = plt.subplots()
    dsvm.plot(fig, ax, log=True, cmap="inferno", title=r"$\rho_{\rm mid}$")
    ax.set_aspect("equal")
    plt.show()
    ```

    <figure markdown>
      ![rhovm](../imgs/rhovm.png){ width="600" }
      <figcaption>Density in the midplane</figcaption>
    </figure>


!!! example "Example 2"

    ```python
    import matplotlib.pyplot as plt
    from nonos.api import GasDataSet
    ds = GasDataSet(43, geometry="polar", directory="tests/data/idefix_planet3d")
    ```

    If now, with the same dataset, we perform a latitudinal projection of the field `RHO`, i.e. the integral of the density between $-\theta$ and $\theta$, and then an azimuthal average, before mapping it in the radial (`"R"`) direction:
    ```python
    dslp = ds["RHO"].latitudinal_projection(theta=3*0.05).azimuthal_average().map("R")
    ```

    Finally, we display the y-label by adding the argument `title`
    ```python
    fig, ax = plt.subplots()
    dslp.plot(fig, ax, c="k", title=r"$\Sigma$")
    plt.show()
    ```

    <figure markdown>
      ![rholpaa](../imgs/rholpaa.png){ width="600" }
      <figcaption>Azimuthally-averaged surface density</figcaption>
    </figure>


!!! example "Example 3"

    As a summary, we show here a simple 2D example.

    ```python
    import matplotlib.pyplot as plt
    from nonos.api import GasDataSet
    ds = GasDataSet(23, directory="tests/data/idefix_newvtk_planet2d")
    fig, ax = plt.subplots()
    ds["VX1"].map("R", "phi").plot(fig, ax, cmap="RdBu_r", title=r"$v_{\rm R}$")
    fig.tight_layout()
    plt.show()
    ```

    <figure markdown>
      ![vx1](../imgs/vx1.png){ width="600" }
      <figcaption>Radial velocity</figcaption>
    </figure>


## Compute a new field

In order to compute a new field from preexisting ones, you can use the `compute` function, which takes 3 mandatory arguments (`field` the name of the new field, `data` the corresponding array and `ref` a known field with similar structure as the new field).

### Example

Let us assume we have a VTK file named `data.0000.vtk` in cartesian geometry.

```python
import numpy as np
from nonos.api import GasDataSet, compute

ds = GasDataSet(0)

Vnorm = compute(
  field="V",
  data=np.sqrt(ds["VX1"].data**2 + ds["VX2"].data**2 + ds["VX3"].data**2),
  ref=ds["VX1"],
)
```

## Coordinates

If `ds` is a dataset, the coordinates `ds.coords` at the cell edges and cell centers can be accessed with the following attributes:

| geometry | cartesian              | polar                    | spherical                    |
|----------|------------------------|--------------------------|------------------------------|
| edges    | (`x`,`y`,`z`)          | (`R`,`phi`,`z`)          | (`r`,`theta`,`phi`)          |
| centers  | (`xmed`,`ymed`,`zmed`) | (`Rmed`,`phimed`,`zmed`) | (`rmed`,`thetamed`,`phimed`) |

## Attributes of fields

If `ds` is a dataset containing the three-dimensional density field `ds["RHO"]`, you can access important quantities, such as:

* `data`: the 3D array.
* `coords`: the coordinates that you can access depending on the geometry.
* `on`: the output number associated with the VTK file.

## Operations on fields

If `ds` is a dataset containing the three-dimensional density field `ds["RHO"]`, several operations on the field are possible.

| API function                         | operation                                | geometry                        |
|--------------------------------------|------------------------------------------|---------------------------------|
| `latitudinal_projection(theta)`      | Integral between $-\theta$ and $\theta$  | `polar`,`spherical`             |
| `vertical_projection(z)`             | Integral between $-z$ and $z$            | `cartesian`,`polar`             |
| `vertical_at_midplane()`             | Slice in the midplane                    | `cartesian`,`polar`,`spherical` |
| `latitudinal_at_theta(theta)`        | Slice at latitude $\theta$               | `polar`,`spherical`             |
| `vertical_at_z(z)`                   | Slice at altitude $z$                    | `cartesian`,`polar`,`spherical` |
| `azimuthal_at_phi(phi)`              | Slice at azimuth $\phi$                  | `polar`,`spherical`             |
| `azimuthal_at_planet(planet_number)` | Slice at planet azimuth $\phi_p$         | `polar`,`spherical`             |
| `azimuthal_average()`                | Azimuthal average                        | `polar`,`spherical`             |
| `radial_at_r(distance)`              | Slice at `distance`                      | `polar`,`spherical`             |
| `radial_average_interval(vmin,vmax)` | Radial average (`vmin` to `vmax`)        | `polar`,`spherical`             |
| `remove_planet_hill(planet_number)`  | Remove the Hill sphere                   | `polar`,`spherical`             |

!!! info "Remove the Hill sphere"

    `remove_planet_hill(planet_number)` removes in the azimuthal direction the contribution between $\phi_p - 2 R_{\rm hill}/R_p$ and $\phi_p + 2 R_{\rm hill}/R_p$ around the planet azimuth $\phi_p$, with $R_{\rm hill}$ the Hill radius and $R_p$ the planet's radial location.

!!! info "Chain the operations"

    Some of these operations can be combined, e.g. first a slice in the midplane and then an azimuthal average with `ds["RHO"].vertical_at_midplane().azimuthal_average()`.

It is also possible to access some other quantities in the arrays:

| API function                 | operation                                          | geometry                        |
|------------------------------|----------------------------------------------------|---------------------------------|
| `find_ir(distance)`          | index in the radial direction at `distance`        | `polar`,`spherical`             |
| `find_imid(altitude)`        | index in the $z$/$\theta$ direction at `altitude`  | `cartesian`,`polar`,`spherical` |
| `find_iphi(phi)`             | index in the azimuthal direction at `phi`          | `polar`,`spherical`             |
| `find_rp(planet_number)`     | radial location of the planet                      | -                               |
| `find_rhill(planet_number)`  | Hill radius of the planet                          | -                               |
| `find_phip(planet_number)`   | azimuthal location of the planet                   | -                               |

!!! info "Additional operations"

    * `map("XDIR","YDIR")`: before plotting the field, we map it in the `("XDIR","YDIR")` plane + optional `planet_corotation` (int) argument to rotate the grid with respect to the corresponding planet number. `("XDIR","YDIR")` can be for example `("R","phi")`, `("x","y")`, `("x","z")`, `("r","theta")`,... depending on the target geometry you want.
    * `diff(on)`: compute the relative difference of the same field for a different VTK file.
    * `save(directory)`: create a .npy file which saves in `directory` the array you just computed.
