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

## Citation

If you use this code, please cite the related paper and this repository. A citation template is provided in `CITATION.cff`.
