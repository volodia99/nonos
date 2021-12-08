from typing import Optional

import numpy as np

from nonos.api.analysis import from_data, from_file


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_around(array, value):
    array = np.asarray(array)
    idx_1 = (np.abs(array - value)).argmin()
    larray = list(array)
    larray.remove(larray[idx_1])
    arraym = np.asarray(larray)
    idx_2 = (np.abs(arraym - value)).argmin()
    return np.asarray([array[idx_1], arraym[idx_2]])


def temporal_all(
    field: str,
    operation: str,
    onall,
    directory: str = "",
    planet_corotation: Optional[int] = None,
):
    datasum = 0
    don = len(onall)
    for on in sorted(onall):
        datafield = from_file(
            field=field, filename=operation, on=on, directory=directory
        )
        datafield_rot = datafield.rotate(planet_corotation=planet_corotation)
        datasum += datafield_rot.data
    datafieldsum = from_data(
        field="".join([field, "T_ALL"]),
        data=np.array(datasum / don),
        coords=datafield.coords,
        on=0,
        operation="_" + operation,
    )
    return datafieldsum


def temporal(
    field: str,
    operation: str,
    onbeg: int,
    *,
    onend: Optional[int] = None,
    directory: str = "",
    planet_corotation: Optional[int] = None,
):
    datasum = 0
    if onend is None:
        datafield = from_file(
            field=field, filename=operation, on=onbeg, directory=directory
        )
        datafield_rot = datafield.rotate(planet_corotation=planet_corotation)
        datafieldsum = from_data(
            field=field,
            data=datafield_rot.data,
            coords=datafield_rot.coords,
            on=onbeg,
            operation="_" + operation,
        )
        return datafieldsum
    else:
        don = onend - onbeg
        for on in range(onbeg, onend + 1):
            datafield = from_file(
                field=field, filename=operation, on=on, directory=directory
            )
            datafield_rot = datafield.rotate(planet_corotation=planet_corotation)
            datasum += datafield_rot.data
        datafieldsum = from_data(
            field="".join([field, f"T_{onbeg}_{onend}"]),
            data=np.array(datasum / don),
            coords=datafield.coords,
            on=onend,
            operation="_" + operation,
        )
        return datafieldsum
