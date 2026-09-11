# Bathymetry migration

This repository computes migrated bathymetry for simple ice melt scenarios. It starts from
present-day ETOPO 2022 topography and ice thickness, removes a chosen amount of Greenland and
Antarctic ice, and solves the sea level equation with [pyslfp](https://github.com/da380/pyslfp)
to find the resulting bedrock, ice and sea surface. The point of the exercise is that sea level
does not change uniformly: the solid Earth deforms under the changing load, the gravity field
changes with it, and shorelines move as a result, so the bathymetry you would map after the melt
is not the original bathymetry shifted by a constant.

The output is a NetCDF file on the original ETOPO grid, so it can be used directly wherever the
present-day grid would have been.

## What the calculation assumes

- The Earth deforms **elastically**. The response is instantaneous, and there is no viscous
  relaxation, so this is not a glacial isostatic adjustment calculation over time.
- The ocean is a **single connected basin**. The Caspian Sea is treated as land, and no other
  inland seas or lakes are modelled separately.
- Shorelines are free to move. The non-linear solve updates the ocean function as sea level and
  the bedrock change, including grounding and ungrounding of ice.
- Rotational feedbacks are included.
- Meltwater comes from Greenland and Antarctica only. Within each of the two regions the
  present-day ice thickness is thinned by a uniform fraction, chosen so that the eustatic
  contribution matches what you asked for. If you ask for more than the ice sheet holds, the melt
  is capped at complete removal and the script tells you what it actually achieved.
- Sea water density is 1028 kg/m³ and ice density is 917 kg/m³.

## Setup

You need Python 3.12 or newer and Poetry.

```bash
poetry install
```

Two notes on the dependencies. pyslfp is installed from a GitHub branch rather than a release,
because the branch adds support for the non-extended grids that native ETOPO data uses; this
becomes an ordinary release dependency once that change is merged. matplotlib is held below 3.11,
where cartopy's labelled gridlines break figure sizing and silently crop panels.

The first run downloads the ETOPO 2022 data from Zenodo into `~/.pyslfp_data` (roughly 200 MB to
download, 450 MB unpacked). Set `PYSLFP_DATA` if you want it somewhere else.

## Running a scenario

```bash
poetry run python bathymetry_migrate.py --gmsl 13 --antarctic-fraction 0.442
```

| Option | Meaning |
| --- | --- |
| `--gmsl` | Target global mean sea level contribution in metres (default 0) |
| `--antarctic-fraction` | Share of the melt taken from Antarctica: 0 is all Greenland, 1 is all Antarctica (default 0) |
| `--resolution` | ETOPO resolution in arc-minutes: 2, 5 or 15 (default 5) |
| `--figures` | Also write diagnostic maps of the solve to `figures/` |

Running with `--gmsl 0` melts nothing and gives the present-day state as the solver sees it. That
file is the natural reference to compare other scenarios against.

To work through a list of scenarios, edit the cases at the top of `run_bathymetry_batch.sh` and
run it:

```bash
./run_bathymetry_batch.sh
RESOLUTION=2 ./run_bathymetry_batch.sh
```

## Outputs

Results go to `migrated_bathymetry/lmax-<lmax>/`, named after the melt that was achieved rather
than the melt that was requested, for example
`bathymetry_gmsl_13p00_frac_0p442_lmax_1079.nc`. The resolution sets the grid and so the maximum
spherical harmonic degree:

| Resolution | lmax | Grid | File size |
| --- | --- | --- | --- |
| 15′ | 359 | 720 × 1440 | 13 MB |
| 5′ | 1079 | 2160 × 4320 | 121 MB |
| 2′ | 2699 | 5400 × 10800 | 758 MB |

Each file holds four fields on the ETOPO `lat`/`lon` grid:

- `bedrock` — bedrock elevation in metres
- `ice_thickness` — ice thickness in metres
- `surface` — elevation of whatever is on top, so ice where there is ice and bedrock otherwise
- `ocean_function` — 1 for ocean, including floating ice shelves, and 0 for land or grounded ice

The scenario is recorded in the file attributes: target sea level contribution, Antarctic
fraction, lmax, resolution and the water density used.

Outputs and figures are not tracked by git.

## Looking at the results

`plot_migrated_sea_level.py` compares two output files and writes a figure to `figures/`. It plots
the surface of each and the sea level change between them, globally and over both poles:

```bash
poetry run python plot_migrated_sea_level.py \
    migrated_bathymetry/lmax-1079/bathymetry_gmsl_0p000_frac_0p00_lmax_1079.nc \
    migrated_bathymetry/lmax-1079/bathymetry_gmsl_13p00_frac_0p442_lmax_1079.nc
```

The difference is always the second file minus the first, so pass the reference run first.
`--cbar-scale` adjusts the sea level colour range if the default saturates.

`verification.ipynb` does the same kind of comparison in more detail over Antarctica and
Greenland, showing bedrock, ice thickness and surface side by side. Set `reference_file` and
`comparison_file` in the first cell and run it.

## Files

| File | Purpose |
| --- | --- |
| `bathymetry_migrate.py` | Runs a scenario end to end and writes the NetCDF output |
| `present_day_topography.py` | Reads the ETOPO files and returns present-day ice thickness and sea level grids for pyslfp |
| `plot_migrated_sea_level.py` | Compares two outputs and plots the sea level change |
| `run_bathymetry_batch.sh` | Runs a list of scenarios one after another |
| `verification.ipynb` | Regional comparison of two outputs |
