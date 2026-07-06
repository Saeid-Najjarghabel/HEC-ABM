# HEC-ABM: Agent-Based Flood Risk and Transport Disruption Model

This repository contains the source code and documentation for a flood-impact assessment framework that couples HEC-RAS 2D hydrodynamic outputs with an agent-based mobility model. The model evaluates dynamic flood exposure, road disruption, route failure, agent-level vulnerability, and risk perception under flood conditions in Kingston upon Thames in London.

## Main features

- Reads HEC-RAS 2D `.hdf` output files and samples flood depth and water-surface elevation.
- Simulates synthetic population agents using age, employment status, gender, travel mode, and trip purpose attributes.
- Generates daily mobility patterns using London Travel Demand Survey-inspired assumptions.
- Uses a Risk Priority Index (RPI) based on flood depth, official warning, and demographic modifiers.
- Simulates risk communication through a Watts-Strogatz small-world social network.
- Dynamically closes flooded road links and recalculates trips using the road graph.
- Calculates road disruption, failed trips, flood exposure, and group-based risk indices.
- Produces figures, heatmaps, snapshots, and scenario outputs for manuscript analysis.

## Repository structure

```text
HEC-ABM/
├── src/
│   ├── flood_abm_model.py          # Main model script
│   └── __init__.py
├── data/
├── examples/
│   └── config_template.yml         # Example configuration file
├── docs/
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── DATA_REQUIREMENTS.md
│   ├── SOFTWARE_AND_DATA_AVAILABILITY.md
│   ├── FAIR_CHECKLIST.md
│   └── REPRODUCIBILITY.md
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
└── .gitignore
```

## Software requirements

The model is written in Python and uses geospatial, network, numerical, and plotting libraries. The hydrodynamic input is generated externally using HEC-RAS 2D.

Core dependencies include:

- Python 3.11 or newer
- NumPy
- pandas
- GeoPandas
- Shapely
- Rasterio
- h5py
- NetworkX
- OSMnx
- PyProj
- Matplotlib
- Seaborn
- SciPy

See `requirements.txt` and `environment.yml` for installation.

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/Flood-ABM.git
cd Flood-ABM
conda env create -f environment.yml
conda activate flood-abm
python src/flood_abm_model.py
```

Before running the model, edit the input paths in `src/flood_abm_model.py` or adapt the script to read from `examples/config_template.yml`.

## Required input data

The current script expects the following local input files:

- HEC-RAS 2D unsteady output file: `.hdf`
- Population points shapefile: `Agents_in_FigureBBox.shp`
- Building footprint/use shapefile: `Buildings_in_FigureBBox.shp`
- Road network data obtained through OSMnx/OpenStreetMap

Public datasets should be cited from their original sources. Processed data used for publication should be deposited in this repository or archived with a DOI using Zenodo, Figshare, or an institutional repository.

## Reproducibility notes

The uploaded script contains hard-coded local Windows paths. For public release, replace private absolute paths with relative paths or configuration variables. A configuration template is provided in `examples/config_template.yml`.

## Citation

If you use this software, please cite the related paper and this repository. A citation template is provided in `CITATION.cff`.
