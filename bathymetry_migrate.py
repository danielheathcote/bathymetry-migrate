"""
Generate a migrated bathymetry map from a non-linear pyslfp sea-level solve.

Given a target eustatic GMSL contribution and a split between Greenland and
Antarctica, this melts the corresponding fraction of present-day ice, solves the
non-linear sea-level equation, and writes the migrated bedrock / ice / surface
fields to a NetCDF file. Optionally produces diagnostic figures.

Example
-------
    python bathymetry_migrate.py --gmsl 5.0 --antarctic-fraction 0.7 --figures
    python bathymetry_migrate.py --gmsl 13 --antarctic-fraction 0.442 --resolution 2
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr

from pyslfp.core import EarthModel, EarthModelParameters
from pyslfp.data import DATADIR, fetch_dataset
from pyslfp.state import EarthState
from pyslfp.physics import SeaLevelEquation

from present_day_topography import get_ice_thickness_and_sea_level

SCRIPT_DIR = Path(__file__).resolve().parent

# ETOPO 2022 bedrock and ice thickness grids, downloaded from Zenodo by pyslfp.
# The folder is read directly rather than via ensure_data("ETOPO"), which looks
# for the wrong folder name until the pyslfp fix/etopo-folder-name PR is merged.
ETOPO_DIR = DATADIR / "pyslfp_etopo"
RESOLUTIONS_ARCMIN = (2, 5, 15)

# Ocean water density (kg/m^3). Matches the existing migrated bathymetry outputs;
# pyslfp's default is 1000.
WATER_DENSITY = 1028.0

BATHYMETRY_DIR = SCRIPT_DIR / "migrated_bathymetry"
FIGURES_DIR = SCRIPT_DIR / "figures"


# --------------------------------------------------------------------------- #
# Filename helpers
# --------------------------------------------------------------------------- #
def _safe_token(token: str) -> str:
    """Keep filenames portable by replacing decimal points and signs."""
    return token.replace("-", "m").replace("+", "").replace(".", "p")


def _sigfig_token(value: float, sigfigs: int) -> str:
    value = float(value)
    if value == 0.0:
        text = "0." + "0" * (sigfigs - 1)
    else:
        decimals = max(sigfigs - 1 - int(np.floor(np.log10(abs(value)))), 0)
        text = f"{value:.{decimals}f}"
    return _safe_token(text)


# --------------------------------------------------------------------------- #
# Input data
# --------------------------------------------------------------------------- #
def etopo_files(resolution: int) -> tuple[Path, Path]:
    """
    Return the ETOPO bedrock and ice thickness files at the given resolution
    (arc-minutes), downloading the ETOPO dataset with pyslfp if it is missing.
    """
    if not ETOPO_DIR.exists() or not any(ETOPO_DIR.iterdir()):
        fetch_dataset("ETOPO")

    topo_path = ETOPO_DIR / f"ETOPO_2022_bedrock_{resolution}.nc"
    ice_path = ETOPO_DIR / f"ETOPO_2022_ice_thickness_{resolution}.nc"
    return topo_path, ice_path


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def implied_gmsl_m_from_ice_change(state: EarthState, ice_change) -> float:
    """Eustatic GMSL contribution (m) from a thickness-change field."""
    direct_load = state.direct_load_from_ice_thickness_change(ice_change)
    gmsl_nd = -state.model.integrate(direct_load) / (
        state.model.parameters.water_density * state.ocean_area
    )
    return gmsl_nd * state.model.parameters.length_scale


def build_initial_state(topo_path: Path, ice_path: Path
                        ) -> tuple[EarthState, EarthModel, SeaLevelEquation]:
    """Construct the present-day Earth state from the high-res topography."""
    params = replace(
        EarthModelParameters.from_defaults(), raw_water_density=WATER_DENSITY
    )
    ice_thickness, sea_level = get_ice_thickness_and_sea_level(
        topo_path,
        ice_path,
        topo_var="elevation",
        ice_var="thickness",
        lon_var="lon",
        lat_var="lat",
        length_scale=params.length_scale,
        water_density=params.raw_water_density,
        ice_density=params.raw_ice_density,
    )

    # The grid is inferred from the native data; EarthModel expects "DH2" to
    # denote a DH grid with sampling=2.
    grid = "DH2" if ice_thickness.sampling == 2 else ice_thickness.grid
    extend = bool(ice_thickness.extend)
    lmax = int(ice_thickness.lmax)
    print(f"Detected grid={grid}, extend={extend}, lmax={lmax} from input grids.")

    earth_model = EarthModel(lmax, parameters=params, grid=grid, extend=extend)
    initial_state = EarthState(
        ice_thickness, sea_level, earth_model, exclude_caspian=True
    )
    sle = SeaLevelEquation(earth_model)
    return initial_state, earth_model, sle


def compute_ice_thickness_change(initial_state: EarthState, target_gmsl_m: float,
                                 antarctic_melt_fraction: float):
    """
    Build the imposed ice-thickness-change field for the requested melt scenario.

    Returns the ice-thickness-change field plus a dict of diagnostics describing
    the achieved (post-capping) eustatic contributions.
    """
    greenland_mask = initial_state.greenland_projection(value=0)
    antarctic_mask = initial_state.antarctic_projection(value=0)

    greenland_full_ice_change = -1.0 * initial_state.ice_thickness * greenland_mask
    antarctic_full_ice_change = -1.0 * initial_state.ice_thickness * antarctic_mask

    greenland_full_gmsl_m = implied_gmsl_m_from_ice_change(
        initial_state, greenland_full_ice_change
    )
    antarctic_full_gmsl_m = implied_gmsl_m_from_ice_change(
        initial_state, antarctic_full_ice_change
    )

    if greenland_full_gmsl_m <= 0.0 or antarctic_full_gmsl_m <= 0.0:
        raise RuntimeError(
            "Could not compute positive full-melt GMSL for Greenland/Antarctica. "
            "Check masks and initial state."
        )

    target_antarctic_gmsl_m = target_gmsl_m * antarctic_melt_fraction
    target_greenland_gmsl_m = target_gmsl_m * (1.0 - antarctic_melt_fraction)

    antarctic_scale_raw = target_antarctic_gmsl_m / antarctic_full_gmsl_m
    greenland_scale_raw = target_greenland_gmsl_m / greenland_full_gmsl_m

    antarctic_scale = min(1.0, antarctic_scale_raw)
    greenland_scale = min(1.0, greenland_scale_raw)

    if antarctic_scale_raw > 1.0:
        warnings.warn(
            "Requested Antarctic contribution exceeds full melt. "
            "Capping Antarctic melt scale at 1.0.",
            RuntimeWarning,
        )
    if greenland_scale_raw > 1.0:
        warnings.warn(
            "Requested Greenland contribution exceeds full melt. "
            "Capping Greenland melt scale at 1.0.",
            RuntimeWarning,
        )

    ice_thickness_change = (
        antarctic_scale * antarctic_full_ice_change
        + greenland_scale * greenland_full_ice_change
    )

    achieved_antarctic_gmsl_m = antarctic_scale * antarctic_full_gmsl_m
    achieved_greenland_gmsl_m = greenland_scale * greenland_full_gmsl_m
    achieved_total_gmsl_m = achieved_antarctic_gmsl_m + achieved_greenland_gmsl_m
    achieved_fraction = (
        achieved_antarctic_gmsl_m / achieved_total_gmsl_m
        if achieved_total_gmsl_m != 0.0
        else 0.0
    )

    diagnostics = {
        "target_total_gmsl_m": target_gmsl_m,
        "achieved_total_gmsl_m": achieved_total_gmsl_m,
        "target_antarctic_gmsl_m": target_antarctic_gmsl_m,
        "achieved_antarctic_gmsl_m": achieved_antarctic_gmsl_m,
        "target_greenland_gmsl_m": target_greenland_gmsl_m,
        "achieved_greenland_gmsl_m": achieved_greenland_gmsl_m,
        "antarctic_scale": antarctic_scale,
        "greenland_scale": greenland_scale,
        "achieved_fraction": achieved_fraction,
    }
    return ice_thickness_change, diagnostics


def print_diagnostics(d: dict) -> None:
    print(f"Target total GMSL (m):        {d['target_total_gmsl_m']:.4f}")
    print(f"Achieved total GMSL (m):      {d['achieved_total_gmsl_m']:.4f}")
    print(f"Target Antarctica GMSL (m):   {d['target_antarctic_gmsl_m']:.4f}")
    print(f"Achieved Antarctica GMSL (m): {d['achieved_antarctic_gmsl_m']:.4f}")
    print(f"Target Greenland GMSL (m):    {d['target_greenland_gmsl_m']:.4f}")
    print(f"Achieved Greenland GMSL (m):  {d['achieved_greenland_gmsl_m']:.4f}")
    print(f"Antarctica scale:             {d['antarctic_scale']:.4f}")
    print(f"Greenland scale:              {d['greenland_scale']:.4f}")


def extract_fields(new_state: EarthState, params) -> dict:
    """Extract dimensional bedrock, ice thickness and surface elevation (m)."""
    bedrock_m = -new_state.sea_level.data * params.length_scale
    ice_thickness_m = new_state.ice_thickness.data * params.length_scale

    rho_w = params.water_density
    rho_i = params.ice_density

    surface_m = np.where(
        new_state.ocean_function.data == 0,
        bedrock_m + ice_thickness_m,
        np.where(
            ice_thickness_m > 0.01,  # guard against ringing
            ice_thickness_m * (1 - rho_i / rho_w),
            bedrock_m,
        ),
    )
    return {
        "bedrock_m": bedrock_m,
        "ice_thickness_m": ice_thickness_m,
        "surface_m": surface_m,
    }


def native_output_coords(path: Path, lat_var: str = "lat",
                         lon_var: str = "lon") -> tuple[np.ndarray, np.ndarray]:
    """
    Return the source ETOPO cell-centre coordinates in the solver's orientation.

    The solver round-trips on a single Driscoll-Healy grid, so output row/column
    i is index-aligned with the native ETOPO sample i. We therefore relabel the
    output onto the original pixel-centre coordinates (lat in N->S order, matching
    `new_state.lats()`; lon as loaded) instead of the DH-node coordinates, which
    are offset by half a cell. See the plan/docstrings for details.
    """
    with xr.open_dataset(path) as ds:
        ds = ds.sortby(lat_var, ascending=False)  # N->S, matches new_state row order
        return ds[lat_var].values, ds[lon_var].values


def save_bathymetry(new_state: EarthState, fields: dict, diagnostics: dict,
                    target_gmsl_m: float, antarctic_melt_fraction: float,
                    lmax: int, topo_path: Path, resolution: int
                    ) -> tuple[xr.Dataset, Path]:
    """Write the migrated bathymetry to NetCDF and return (dataset, path)."""
    output_dir = BATHYMETRY_DIR / f"lmax-{lmax}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Relabel onto the source ETOPO cell-centre grid (the data is index-aligned
    # with the native samples) rather than the half-cell-offset DH-node grid.
    lat_coord, lon_coord = native_output_coords(topo_path)
    if len(lat_coord) != len(new_state.lats()) or len(lon_coord) != len(new_state.lons()):
        raise ValueError(
            f"Source ETOPO grid ({len(lat_coord)}x{len(lon_coord)}) does not match the "
            f"solver grid ({len(new_state.lats())}x{len(new_state.lons())}); cannot relabel."
        )

    ds = xr.Dataset(
        data_vars={
            "bedrock": (("lat", "lon"), fields["bedrock_m"].astype("float32")),
            "ice_thickness": (("lat", "lon"), fields["ice_thickness_m"].astype("float32")),
            "surface": (("lat", "lon"), fields["surface_m"].astype("float32")),
            "ocean_function": (
                ("lat", "lon"),
                new_state.ocean_function.data.astype("int8"),
            ),
        },
        coords={
            "lat": lat_coord,
            "lon": lon_coord,
        },
        attrs={
            "description": "Migrated bathymetry map from non-linear pyslfp solve",
            "coordinate_note": (
                "coords are source ETOPO pixel centres; data computed on DH nodes "
                "(~half-cell offset internally, negligible for the smooth solve)"
            ),
            "target_gmsl_m": float(target_gmsl_m),
            "antarctic_melt_fraction": float(antarctic_melt_fraction),
            "lmax": int(lmax),
            "resolution_arcmin": int(resolution),
            "water_density": float(WATER_DENSITY),
        },
    )

    ds["bedrock"].attrs.update(long_name="Migrated bedrock elevation", units="m")
    ds["ice_thickness"].attrs.update(long_name="Migrated ice thickness", units="m")
    ds["surface"].attrs.update(
        long_name="Migrated surface elevation (height of highest point of ice or bedrock)",
        units="m",
    )
    ds["ocean_function"].attrs.update(
        long_name="Ocean function (1 = ocean incl. floating ice shelves, 0 = land/grounded ice)",
        flag_values="0, 1",
        flag_meanings="land_or_grounded_ice ocean_or_ice_shelf",
        note="Caspian Sea excluded (treated as land)",
    )

    # Flip latitude to South-to-North (ascending) before saving.
    ds = ds.sortby("lat")

    gmsl_token = _sigfig_token(diagnostics["achieved_total_gmsl_m"], 4)
    frac_token = _sigfig_token(diagnostics["achieved_fraction"], 3)
    path = output_dir / f"bathymetry_gmsl_{gmsl_token}_frac_{frac_token}_lmax_{lmax}.nc"
    ds.to_netcdf(path)

    print(f"Saved migrated bathymetry to: {path}")
    print(f"Max depth found: {np.min(ds['bedrock'].values):.2f} m")
    return ds, path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(initial_state, new_state, params, ice_thickness_change,
                 slc_nonlin, disp, potc, bathymetry_ds, gmsl_token, frac_token,
                 lmax) -> None:
    """Produce and save the diagnostic figures into FIGURES_DIR."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    from pyslfp.plot import plot

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"gmsl_{gmsl_token}_frac_{frac_token}_lmax_{lmax}"

    direct_ice_load_dimensional = (
        initial_state.direct_load_from_ice_thickness_change(ice_thickness_change)
        * params.load_scale
    )

    # --- Figure 1: solve diagnostics (load / slc / disp / geoid) ---------- #
    fig, ax_dict = plt.subplot_mosaic(
        [["load", "slc"], ["disp", "potc"]],
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )

    plot(
        direct_ice_load_dimensional,
        ax=ax_dict["load"],
        coasts=False,
        symmetric=True,
        colorbar_kwargs={"label": "Direct ice load change (kg m$^{-2}$)"},
    )
    ax_dict["load"].set_title("Full imposed ice load")
    initial_state.plot_coastline(ax_dict["load"], color="black", linewidth=0.8)

    masked_slc = slc_nonlin * new_state.ocean_projection() * params.length_scale
    slc_vmax = float(np.nanmax(masked_slc.data))
    plot(
        masked_slc,
        ax=ax_dict["slc"],
        coasts=False,
        vmin=-slc_vmax * 2,
        vmax=slc_vmax * 2,
        colorbar_kwargs={"label": "Sea level change (m)"},
    )
    ax_dict["slc"].set_title("Non-linear sea-level fingerprint (ocean only)")
    new_state.plot_coastline(ax_dict["slc"], color="red", linewidth=0.8)

    plot(
        disp * params.length_scale,
        ax=ax_dict["disp"],
        coasts=False,
        symmetric=True,
        colorbar_kwargs={"label": "Vertical displacement (m)"},
    )
    ax_dict["disp"].set_title("Vertical displacement")
    new_state.plot_coastline(ax_dict["disp"], color="red", linewidth=0.8)

    plot(
        (-1.0 * potc) * params.length_scale / params.gravitational_acceleration,
        ax=ax_dict["potc"],
        coasts=False,
        symmetric=True,
        colorbar_kwargs={"label": "Geoid anomaly (m)"},
    )
    ax_dict["potc"].set_title("Geoid anomaly")
    new_state.plot_coastline(ax_dict["potc"], color="red", linewidth=0.8)

    fig.savefig(FIGURES_DIR / f"solve_diagnostics_{suffix}.png", dpi=150)

    # --- Figure 2: migrated vs original bathymetry ------------------------ #
    new = new_state.ocean_function * new_state.sea_level * params.length_scale
    old = initial_state.ocean_function * initial_state.sea_level * params.length_scale
    diff = new - old

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Difference between migrated bathymetry and original ETOPO", fontsize=16)
    vmax_val = slc_vmax * 3
    vmin_val = -slc_vmax * 3

    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.Robinson())
    plot(
        diff,
        ax=ax1,
        coasts=False,
        vmax=vmax_val,
        vmin=vmin_val,
        colorbar_kwargs={
            "label": "Bathymetry change (m)",
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    new_state.plot_coastline(ax1, color="red", linewidth=0.8)
    ax1.set_title("Global")

    ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.NorthPolarStereo(central_longitude=-45))
    plot(
        diff,
        ax=ax2,
        coasts=False,
        vmax=vmax_val,
        vmin=vmin_val,
        map_extent=[-75, -10, 55, 85],
        colorbar=False,
    )
    new_state.plot_coastline(ax2, color="red", linewidth=0.8)
    ax2.set_title("Greenland Extent")

    ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.SouthPolarStereo())
    plot(
        diff,
        ax=ax3,
        coasts=False,
        vmax=vmax_val,
        vmin=vmin_val,
        map_extent=[-180, 180, -90, -60],
        colorbar=False,
    )
    new_state.plot_coastline(ax3, color="red", linewidth=0.8)
    ax3.set_title("Antarctica")

    fig.savefig(FIGURES_DIR / f"bathymetry_diff_{suffix}.png", dpi=150)

    # --- Figure 3: surface topography (global + north polar) -------------- #
    ds = bathymetry_ds
    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    im1 = ax1.pcolormesh(
        ds.lon, ds.lat, ds.surface,
        transform=ccrs.PlateCarree(), cmap="terrain", shading="auto",
    )
    ax1.set_global()
    ax1.set_title("(a) Surface topography (m)")
    plt.colorbar(im1, ax=ax1, shrink=0.6, orientation="horizontal", pad=0.05)

    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.NorthPolarStereo())
    ax2.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    im2 = ax2.pcolormesh(
        ds.lon, ds.lat, ds.surface,
        transform=ccrs.PlateCarree(), cmap="terrain", shading="auto",
    )
    ax2.coastlines()
    ax2.set_title("(b) Surface topography (m)")
    gl = ax2.gridlines(linestyle="--", draw_labels=False, x_inline=False, y_inline=False)
    gl.xlocator = mticker.MultipleLocator(30)
    gl.ylocator = mticker.MultipleLocator(30)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    plt.colorbar(im2, ax=ax2, shrink=0.6, orientation="horizontal", pad=0.05)

    fig.savefig(FIGURES_DIR / f"surface_topography_{suffix}.png", dpi=150)

    # --- Figure 4: Bering Strait surface topography ----------------------- #
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([-173, -165, 62, 70], crs=ccrs.PlateCarree())
    im = ax.pcolormesh(
        ds.lon, ds.lat, ds.surface,
        transform=ccrs.PlateCarree(), cmap="terrain", shading="auto",
        vmin=-100, vmax=100,
    )
    ax.coastlines(color="white", linewidth=1)
    ax.set_title("Bering Strait Surface Topography (m)")
    gl = ax.gridlines(linestyle="--", draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(2)
    gl.ylocator = mticker.MultipleLocator(2)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    plt.colorbar(im, ax=ax, shrink=0.7, orientation="horizontal", pad=0.08, label="Depth (m)")

    fig.savefig(FIGURES_DIR / f"bering_strait_{suffix}.png", dpi=150)

    print(f"Saved figures to: {FIGURES_DIR}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a migrated bathymetry map from a non-linear pyslfp solve.",
    )
    parser.add_argument(
        "--gmsl", type=float, default=0.0,
        help="Target global mean sea-level contribution (m) from Greenland + Antarctica.",
    )
    parser.add_argument(
        "--antarctic-fraction", type=float, default=0.0,
        help="Fraction of the melt from Antarctica (0 = all Greenland, 1 = all Antarctica).",
    )
    parser.add_argument(
        "--resolution", type=int, choices=RESOLUTIONS_ARCMIN, default=5,
        help="Resolution of the input ETOPO grids in arc-minutes (default: 5).",
    )
    parser.add_argument(
        "--figures", action="store_true",
        help="If set, plot diagnostic figures and save them to the figures/ folder.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.antarctic_fraction <= 1.0):
        parser.error("--antarctic-fraction must lie in [0, 1].")

    topo_path, ice_path = etopo_files(args.resolution)

    initial_state, earth_model, sle = build_initial_state(topo_path, ice_path)
    params = earth_model.parameters
    lmax = earth_model.lmax

    ice_thickness_change, diagnostics = compute_ice_thickness_change(
        initial_state, args.gmsl, args.antarctic_fraction
    )
    print_diagnostics(diagnostics)

    new_state, slc_nonlin, disp, potc, avc = sle.solve_nonlinear_equation(
        initial_state,
        ice_thickness_change=ice_thickness_change,
    )

    fields = extract_fields(new_state, params)
    bathymetry_ds, _ = save_bathymetry(
        new_state, fields, diagnostics, args.gmsl, args.antarctic_fraction, lmax,
        topo_path, args.resolution,
    )

    if args.figures:
        gmsl_token = _sigfig_token(diagnostics["achieved_total_gmsl_m"], 4)
        frac_token = _sigfig_token(diagnostics["achieved_fraction"], 3)
        make_figures(
            initial_state, new_state, params, ice_thickness_change,
            slc_nonlin, disp, potc, bathymetry_ds, gmsl_token, frac_token, lmax,
        )


if __name__ == "__main__":
    main()
