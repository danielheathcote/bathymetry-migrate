import argparse
import sys
import re
from pathlib import Path

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as cmo

def plot_map(fig, ax, data, cmap, title, label, vmin=None, vmax=None, extent=None):
    """Stock panel plotter: shared boilerplate for every map."""
    if extent is not None:
        ax.set_extent(list(extent), ccrs.PlateCarree())
    else:
        ax.set_global()

    im = data.plot.pcolormesh(
        ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False,
        rasterized=True # Speeds up rendering/saving for high-res grids
    )

    # Polar projections can sometimes throw warnings with grid labels; we'll keep it safe
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    # Hide top/right labels to avoid clutter
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label=label)
    return im

def get_domain_sym_limit(da, extent=None, scale=1.0):
    """
    Calculates a symmetric limit based on the maximum absolute value
    within a given spatial extent (lon_min, lon_max, lat_min, lat_max).
    Multiplies the max by `scale`.
    """
    if extent is None:
        max_val = float(np.nanmax(np.abs(da)))
    else:
        _, _, lat_min, lat_max = extent
        # Mask out values outside the latitude band to find the regional max
        # (ignoring longitude filtering since circumpolar projections show all longitudes)
        subset = da.where((da.lat >= lat_min) & (da.lat <= lat_max))
        max_val = float(np.nanmax(np.abs(subset)))

    return max_val * scale

def parse_title_from_filename(filename_stem):
    """Parses gmsl, frac, and lmax from the filename to create a readable title."""
    gmsl_str = "Unknown"
    frac_str = "Unknown"
    lmax_str = "Unknown"
    res_str = "Unknown"

    # Extract GMSL
    m_gmsl = re.search(r'gmsl_([0-9]+p[0-9]+)', filename_stem)
    if m_gmsl:
        gmsl_str = m_gmsl.group(1).replace('p', '.')

    # Extract fraction
    m_frac = re.search(r'frac_([0-9]+p[0-9]+)', filename_stem)
    if m_frac:
        frac_str = m_frac.group(1).replace('p', '.')

    # Extract lmax and calculate resolution
    m_lmax = re.search(r'lmax_([0-9]+)', filename_stem)
    if m_lmax:
        lmax_str = m_lmax.group(1)
        lmax_val = int(lmax_str)
        if lmax_val > 0:
            # Nyquist spatial resolution for spherical harmonics:
            # N_lat = 2 * (Lmax + 1)
            # Resolution = 180 / N_lat
            res_deg = 180.0 / (2 * (lmax_val + 1))
            res_arcmin = res_deg * 60

            # Format nicely as a fraction if it's a common one (like 1/12)
            # or just default to decimals if it's not a clean fraction
            if np.isclose(res_arcmin, round(res_arcmin)):
                res_str = f"1/{int(60/round(res_arcmin))}° ({round(res_arcmin)}')".replace("1/1°", "1°")
            else:
                res_str = f"{res_deg:.3f}° (~{res_arcmin:.1f}')"

    title = (f"Global mean sea-level change: {gmsl_str}m. "
             f"Antarctic fraction: {frac_str}. "
             f"Lmax: {lmax_str}. "
             f"Resolution: {res_str}")
    return title

def short_stem(path):
    """Returns the filename stem without the leading 'bathymetry_' prefix."""
    stem = path.stem
    return stem[len("bathymetry_"):] if stem.startswith("bathymetry_") else stem

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Plot surfaces and sea level change between a reference and a comparison migrated dataset.")
    parser.add_argument("reference_file", type=str, help="Path to the reference migrated dataset NetCDF file (e.g. a gmsl 0 run)")
    parser.add_argument("comparison_file", type=str, help="Path to the comparison migrated dataset NetCDF file")
    parser.add_argument("--cbar-scale", type=float, default=1.0,
                        help="Scaling factor for the sea level change colorbar maximum (default: 1.0)")
    args = parser.parse_args()

    reference_path = Path(args.reference_file)
    comparison_path = Path(args.comparison_file)
    for name, path in (("reference", reference_path), ("comparison", comparison_path)):
        if not path.exists():
            print(f"Error: Could not find {name} file at {path}")
            sys.exit(1)

    print(f"Loading datasets...")
    reference = xr.open_dataset(reference_path)
    bed_r = reference['bedrock']
    sur_r = reference['surface']

    comparison = xr.open_dataset(comparison_path)
    bed_c = comparison['bedrock']
    sur_c = comparison['surface']

    # Load ocean function for masking (1 = ocean, 0 = land)
    ocn_func_c = comparison['ocean_function']

    # Check whether the reference and comparison datasets share the same lat/lon grid
    coords_match = (
        np.array_equal(reference.lat.values, comparison.lat.values) and
        np.array_equal(reference.lon.values, comparison.lon.values)
    )

    if not coords_match:
        print("Error: Lat/lon grids differ between reference and comparison datasets. Cannot compute difference.")
        sys.exit(1)

    # Calculate Sea Level Change
    # Sea level is the negative of bedrock, so SLC is (-comparison_bedrock) - (-reference_bedrock)
    slc_unmasked = (-bed_c) - (-bed_r)

    # Mask out land so we only see sea level change over the ocean
    slc = slc_unmasked.where(ocn_func_c == 1)

    print("Generating plots...")
    # Setup the figure, slightly wider to accommodate the new ratio comfortably.
    # Constrained layout rather than tight_layout: with matplotlib 3.11, tight_layout
    # breaks cartopy's labelled gridlines on the polar projections.
    fig = plt.figure(figsize=(24, 20), layout='constrained')

    # Generate and set the super title
    super_title = (f"Comparison: {parse_title_from_filename(comparison_path.stem)}\n"
                   f"Reference: {parse_title_from_filename(reference_path.stem)}")
    fig.suptitle(super_title, fontsize=20, fontweight='bold')

    # Define columns: [Projection, Cartopy Extent, Column Title suffix]
    columns = [
        (ccrs.Robinson(), None, "(Global)"),
        (ccrs.NorthPolarStereo(), [-180, 180, 60, 90], "(North Pole)"),
        (ccrs.SouthPolarStereo(), [-180, 180, -90, -60], "(South Pole)")
    ]

    # Define rows: [Data, Colormap, Base Title, Colorbar Label, Calculate Dynamic Limits]
    rows = [
        (sur_r, cmo.cm.topo, "Reference Surface", "Elevation (m)", False),
        (sur_c, cmo.cm.topo, "Comparison Surface", "Elevation (m)", False),
        (slc, cmo.cm.balance, "Sea Level Change (Comparison - Reference)", "SLC (m)", True)
    ]

    # Set up a GridSpec with customized width ratios
    gs = fig.add_gridspec(3, 3, width_ratios=[1.4, 1.0, 1.0])

    # Plot everything
    for r, (data, cmap, base_title, label, dynamic_limits) in enumerate(rows):
        for c, (proj, extent, col_title) in enumerate(columns):
            # Use the GridSpec indices to create the subplot
            ax = fig.add_subplot(gs[r, c], projection=proj)
            full_title = f"{base_title}\n{col_title}"

            # Determine colorbar limits for this specific subplot
            if dynamic_limits:
                lim = get_domain_sym_limit(data, extent=extent, scale=args.cbar_scale)
                vmin, vmax = -lim, lim
            else:
                vmin, vmax = None, None

            plot_map(
                fig, ax, data, cmap, full_title, label,
                vmin=vmin, vmax=vmax, extent=extent
            )

    # Create output directory
    script_dir = Path(__file__).resolve().parent
    fig_dir = script_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Setup the output filename dynamically based on both input file names
    out_name = f"sea-level-change_{short_stem(comparison_path)}_vs_{short_stem(reference_path)}"
    out_path = fig_dir / f"{out_name}.png"

    print(f"Saving figure to {out_path} ...")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print("Done!")

if __name__ == '__main__':
    main()
