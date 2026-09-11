"""
Module for loading static, high-resolution present-day (0 ka) topography
and ice thickness data as pyslfp-ready grids.
"""

from __future__ import annotations
from typing import Tuple, Union
from pathlib import Path

import numpy as np
import xarray as xr
from pyshtools import SHGrid

from pyslfp.core import WATER_DENSITY, ICE_DENSITY


def get_ice_thickness_and_sea_level(
    topo_file: Union[str, Path],
    ice_file: Union[str, Path],
    /,
    *,
    topo_var: str = "elevation",
    ice_var: str = "thickness",
    lon_var: str = "lon",
    lat_var: str = "lat",
    length_scale: float = 1.0,
    water_density: float = WATER_DENSITY,
    ice_density: float = ICE_DENSITY,
    grid: str = "DH",
) -> Tuple[SHGrid, SHGrid]:
    """
    Returns the scaled ice thickness and sea level for the present-day (0 ka)
    Earth, using two high-resolution NetCDF files (one for bedrock, one for ice
    thickness) and calculating flotation.

    The grids are built directly from the native arrays, so their lmax, sampling
    and extend state are inferred from the array shape. To avoid spatial aliasing
    (ringing), build the EarthModel to match the returned grids, e.g.
    ``EarthModel(ice.lmax, grid="DH2" if ice.sampling == 2 else ice.grid,
    extend=ice.extend)``.

    Note on registration: the native arrays are fed directly to
    ``SHGrid.from_array`` without resampling. ETOPO data is pixel/cell-centre
    registered, whereas a Driscoll-Healy grid is node/gridline registered (row 0
    on +90 deg, column 0 on lon 0). The values are therefore relabelled onto the
    DH nodes, i.e. shifted by half a grid cell (~0.04 deg at lmax=1079). This is
    negligible for the smooth global solve; downstream products that need to align
    with the source ETOPO should relabel coordinates back to the pixel centres.

    Args:
        topo_file (str | Path): Path to the high-res bedrock NetCDF file.
        ice_file (str | Path): Path to the high-res ice thickness NetCDF file.
        topo_var (str): The variable name for topography.
        ice_var (str): The variable name for ice thickness.
        lon_var (str): The variable name for longitude in both files.
        lat_var (str): The variable name for latitude in both files.
        length_scale (float): Scaling factor to non-dimensionalize outputs.
        water_density (float): Ocean water density (kg/m^3) for the grounding
            test. Defaults to the package constant; pass the EarthModel's
            ``raw_water_density`` to keep the flotation logic consistent.
        ice_density (float): Ice density (kg/m^3) for the grounding test.
            Defaults to the package constant; pass the EarthModel's
            ``raw_ice_density`` to stay consistent.
        grid (str): The pyshtools grid format (e.g., "DH").

    Returns:
        Tuple[SHGrid, SHGrid]: The (ice_thickness, sea_level) grids,
        scaled by `length_scale`.
    """
    ice_thickness, topography = _get_ice_thickness_and_topography(
        topo_file,
        ice_file,
        topo_var=topo_var,
        ice_var=ice_var,
        lon_var=lon_var,
        lat_var=lat_var,
        length_scale=length_scale,
        water_density=water_density,
        ice_density=ice_density,
        grid=grid,
    )

    sea_level = SHGrid.from_array(
        np.where(
            topography.data < 0,
            -topography.data,
            -topography.data + ice_thickness.data,
        ),
        grid=grid,
    )

    return ice_thickness, sea_level


def _get_ice_thickness_and_topography(
    topo_file: Union[str, Path],
    ice_file: Union[str, Path],
    /,
    *,
    topo_var: str,
    ice_var: str,
    lon_var: str,
    lat_var: str,
    length_scale: float,
    water_density: float,
    ice_density: float,
    grid: str,
) -> Tuple[SHGrid, SHGrid]:
    """
    Returns the scaled ice thickness and bedrock topography.
    """
    topo_file = Path(topo_file)
    ice_file = Path(ice_file)

    if not topo_file.exists():
        raise FileNotFoundError(f"Topography data file not found: {topo_file}")
    if not ice_file.exists():
        raise FileNotFoundError(f"Ice data file not found: {ice_file}")

    print(f"Loading high-resolution bedrock from {topo_file.name}...")
    topo_data = xr.open_dataset(topo_file)
    # Flip latitude to be North-to-South (descending)
    topo_data = topo_data.sortby(lat_var, ascending=False)
    # Transpose to ensure (lat, lon) order
    topo_data = topo_data.transpose(lat_var, lon_var)
    topo_array = topo_data[topo_var].values

    print(f"Loading high-resolution ice thickness from {ice_file.name}...")
    ice_data = xr.open_dataset(ice_file)
    # Apply the exact same flip to the ice data
    ice_data = ice_data.sortby(lat_var, ascending=False)
    ice_data = ice_data.transpose(lat_var, lon_var)

    # Replace NaNs over ocean/land safely to 0 meters of ice
    ice_array = np.nan_to_num(ice_data[ice_var].values, nan=0.0)

    # Scale the native arrays directly
    topo_scaled = topo_array / length_scale
    ice_scaled = ice_array / length_scale

    # Determine where the ice is firmly grounded on the solid earth.
    rho_i = ice_density
    rho_w = water_density
    is_grounded = (topo_scaled > 0) | (rho_w * -topo_scaled <= rho_i * ice_scaled)

    # Combine grounded ice with topography
    topo_combined = np.where(
        is_grounded,
        topo_scaled + ice_scaled,
        topo_scaled,
    )

    # Construct PySHTools Grids directly from the native arrays. The grid type
    # (lmax, sampling and extend state) is inferred from the array shape by from_array.
    topo_grid = SHGrid.from_array(topo_combined, grid=grid)
    ice_grid = SHGrid.from_array(ice_scaled, grid=grid)

    # Validate that both files describe the same grid. A mismatch here means the
    # topography and ice data were produced at different resolutions.
    for attr in ("lmax", "sampling", "extend"):
        topo_value = getattr(topo_grid, attr, None)
        ice_value = getattr(ice_grid, attr, None)
        if topo_value != ice_value:
            raise ValueError(
                f"Topography grid {attr} ({topo_value}) does not match ice grid "
                f"{attr} ({ice_value}). Both files must share the same grid."
            )

    return ice_grid, topo_grid
