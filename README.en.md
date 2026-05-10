# Biomechanical Analysis of Lateral Perturbations During Walking

[中文版本](./README.md) | [English Version](./README.en.md)

## Overview

This project investigates the **kinematic and kinetic effects of lateral pushing perturbations applied during normal walking**. Using a custom-built instrumented boxing pad, lateral pushes of varying intensities are delivered to subjects at specific gait events, while motion capture (Qualisys), force plate (Kistler), and pad-mounted force sensor data are synchronously recorded. The data pipeline covers raw signal processing, musculoskeletal modeling (OpenSim inverse kinematics & dynamics), statistical analysis, and visualization.

## Project Structure

### Configuration & Entry Point

| File | Description |
|------|-------------|
| `main.py` | Pipeline entry point. Defines subject metadata and iterates through all subjects to initiate the `analyze_subject()` workflow. |
| `measuresheet.ipynb` | Jupyter notebook that generates randomized trial lists for experiments, controlling the order and intensity of push conditions. |

### Core Modules

| File | Description |
|------|-------------|
| `basic.py` | **Core signal processing library.** Includes `VoltConverter` (raw voltage → force/moment conversion), `spectral_subtraction` / `spectral_subtraction_same_trial` (walking noise removal), `Mask` (impact gating & silence detection), `Filter` (Butterworth, median, wavelet, notch, etc.), `transform_pad_to_global` (coordinate system transformation), and other utility functions. |
| `subject.py` | **Subject-level data management.** `Subject` class handles trial list reading, mocap/pad data loading, batch processing of all trials, and result aggregation. |
| `trial.py` | **Trial-level analysis pipeline.** `Trial` class coordinates data offset removal, filtering, event-based cropping, gait cycle normalization, OpenSim IK/ID solving, and extracting kinematic/kinetic outcome measures (velocity, push impulse, joint moments, etc.). |
| `file.py` | **File I/O module.** Provides `.trc` / `.sto` export for OpenSim, `.c3d` reading via subprocess, pad data reading and preprocessing (voltage conversion, coordinate transformation), and `.mot` / `.sto` parsing utilities. |

### Data Reading

| File | Description |
|------|-------------|
| `readc3d.py` | `C3DReader` class — reads `.c3d` files via a persistent subprocess worker to circumvent DLL conflicts between `ezc3d` and `opensim` loaded in the same process. |
| `readc3d_subprocess.py` | The subprocess worker that uses `ezc3d` to parse `.c3d` files and returns data via pickle serialization over stdin/stdout. |

### QTM Processing

| File | Description |
|------|-------------|
| `qtm_process.py` | Script for the **Qualisys QTM Python scripting interface**. Automates gait event detection (heel strikes via force threshold + heel marker minima), timeline cropping, 3D tracking, gap filling, AIM (Automatic Identification of Markers), and configuration of C3D export and file save — both single-trial and batch modes. |

### Statistics & Visualization

| File | Description |
|------|-------------|
| `stats.py` | **Statistical modeling.** Fits Linear Mixed Effects Model (LMM) to analyze effects of walking speed and push impulse on displacement, velocity change and joint moments. Generates individual-slope plots, Q-Q diagnostics, and exports summary tables. |
| `descriptive_stats.py` | Generates individual boxplots for baseline speed, push impulse, and peak push force across speed/intensity conditions, including pooled data overlay. |
| `descriptive_stats_subplots.py` | Generates a combined horizontal subplot figure of the three descriptive boxplots in a single image. |

### Utility & Miscellaneous

| File | Description |
|------|-------------|
| `test.py` | Ad-hoc testing script for orientation of pad coordinate system. |
| `tmp.py` | Temporary utility (e.g., batch file renaming). |

### Legacy

| File | Description |
|------|-------------|
| `MATLAB/voltToMechanics.m` | MATLAB prototype for volt-to-mechanics conversion. |
| `MATLAB/getSpeedTrialInd.m` | MATLAB utility to extract speed and trial index from filenames. |
| `MATLAB/testDataInspection.m` | MATLAB test script for data inspection. |

## Dependencies

- Python 3.11+
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- `pywt` (wavelet denoising)
- `sympy` (symbolic computation)
- `opensim` (musculoskeletal simulation — IK/ID)
- `ezc3d` (C3D file parsing, used in subprocess)
- `statsmodels` (linear mixed models)
- `qtm` (Qualisys QTM scripting API)

## Pipeline Workflow

1. **Experiment design** → `measuresheet.ipynb` generates randomized trial lists.
2. **Data acquisition** → Qualisys (motion capture), Kistler force plates, and the instrumented boxing pad record synchronously.
3. **QTM processing** → `qtm_process.py` detects gait events, tracks markers, fills gaps, and exports C3D.
4. **Data reading** → `readc3d.py` + `readc3d_subprocess.py` parse C3D files. `file.py` reads pad voltage data.
5. **Signal processing** → `basic.py` converts voltages to forces, removes walking noise via spectral subtraction, gates push periods, and filters all signals.
6. **Analysis** → `subject.py` + `trial.py` orchestrate per-subject/per-trial pipelines including OpenSim IK/ID.
7. **Statistics & visualization** → `stats.py`, `descriptive_stats.py`, and `descriptive_stats_subplots.py` produce LMM results and easily readable figures.

## Notes

- All file paths in the scripts are absolute and point to specific local directories. Adjust `DATA_DIR`, `ANALYSIS_DIR`, and related paths before running on a different machine.
- `readc3d.py` uses a subprocess architecture to avoid the `ezc3d` / `opensim` DLL conflict — the worker process persists across requests for efficiency.
- Trial lists (`.csv`) define the mapping between recorded file indices and experimental conditions.
