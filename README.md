# Immune Cell Migration

A Python package for automated tracking and analysis of immune cell migration in microscopy images. Uses deep learning (U-Net) to detect cells, applies sophisticated tracking algorithms, and generates detailed analysis of cell motility, speed, persistence, and chemotactic response.

## Features

- **Cell Detection**: Pre-trained U-Net deep learning model for segmenting immune cells (NK cells, K562 cells, Jurkat cells, pigPBMCs) in 3D microscopy data
- **Cell Tracking**: Overlap-based tracking with greedy stitching algorithm to connect cells across frames
- **Motion Analysis**: Calculates motile fraction, speed (multiple methods), persistence, and chemotactic response
- **Data Processing**: Drift correction, ClickPoints database preparation, and Excel export
- **Visualization**: Publication-quality plots including KDE distributions, bar plots, and quadrant analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/fabrylab/immune_cell_migration.git
cd immune_cell_migration

# Install the package with dependencies
pip install -e .
```

### Dependencies

**Core Libraries:**
- numpy (<2.0), scipy, scikit-image, pandas, matplotlib, seaborn

**Deep Learning:**
- tensorflow (==2.15), keras

**Image Processing:**
- tifffile, opencv-python, clickpoints

**Data Management:**
- peewee, xlsxwriter, openpyxl, joblib, tqdm, natsort

## Usage

### Standard Migration Assay

```python
from immune_cell_migration.pipelines import migration_assay

# Run complete migration analysis pipeline
migration_assay.complete_pipeline(
    folder="/path/to/data",
    time_step=15,                              # seconds between frames
    conditions=["control", "treatment"],       # experimental conditions
    pos_num=4,                                 # positions per condition
    celltype="NK",                             # cell type for detection
    acq_mode="sequential",                     # acquisition mode
    savename="experiment_001",                 # output file prefix
    order=["control", "treatment"],            # order for plotting
    drift_corr=True,                           # apply drift correction
    clickpoints_db=True,                       # create ClickPoints databases
    tracking=True,                             # run cell tracking
    postprocessing=True,                       # calculate metrics
    plotting=True,                             # generate plots
    n_jobs=4                                   # parallel processing cores
)
```

### Chemokine Assay

For analyzing directional migration toward chemokine gradients:

```python
from immune_cell_migration.pipelines import chemokine_assay

chemokine_assay.complete_pipeline(
    folder="/path/to/data",
    time_step=15,
    conditions=["control", "chemokine"],
    pos_num=4,
    celltype="NK",
    acq_mode="sequential",
    savename="chemokine_experiment",
    order=["control", "chemokine"],
    chem_dir="up",                             # chemokine gradient direction
    drift_corr=True,
    clickpoints_db=True,
    tracking=True,
    postprocessing=True,
    plotting=True
)
```

## Workflow Overview

### 1. Preprocessing
- Raw TIFF image sequences are processed with optional drift correction
- Images are organized into ClickPoints databases with multiple layers (MaxProj, MinProj, MaxIndices, MinIndices)

### 2. Cell Detection
- U-Net deep learning model detects cell positions
- Creates segmentation masks stored in the ClickPoints database

### 3. Cell Tracking
- Overlap-based tracking connects cells frame-to-frame
- Greedy stitching algorithm merges fragmented tracks
- Parameters optimized for different cell types

### 4. Postprocessing & Analysis
- **Speed**: Calculated via bounding box and step-width methods (um/min)
- **Persistence**: Mean cosine angle of movement direction
- **Motility Classification**: Cells classified as motile/non-motile based on cell-type-specific thresholds
- Results exported to CSV and Excel files

### 5. Visualization
- KDE distributions of speed and persistence
- Bar plots comparing conditions
- Directional analysis and quadrant plots

## Supported Cell Types

| Cell Type | Motility Threshold (um/min) |
|-----------|---------------------------|
| NK | 6.5 |
| NK_day14 | 13.0 |
| pigPBMCs | 6.0 |
| Jurkat | 4.0 |

## Data Input Format

**Image Requirements:**
- TIFF format image sequences
- Multiple layers (minimum- and maximum-intensity projections, plus z-information)
- File naming convention: `{YYYYMMDD}-{HHMMSS}_{rep}_{pos}_{mode}_z*.tif`

**Supported Cameras:**
- Lumenera: 4.56 um/pixel
- Basler: 3.45 um/pixel
- Hamamatsu: 6.45 um/pixel

## Output

**Analysis Results:**
- Excel files (.xlsx) with aggregated metrics per condition
- CSV files with per-track motion metrics
- PNG/PDF plots

**Metrics Exported:**
- Motile fraction (%)
- Speed (um/min) with multiple calculation methods
- Persistence (directionality measure)
- Persistent fraction (%)
- Track counts (total, motile)
- Directional metrics (for chemokine assays)

## Project Structure

```
immune_cell_migration/
├── immune_cell_migration/
│   ├── pipelines/              # High-level analysis workflows
│   │   ├── migration_assay.py  # Standard migration pipeline
│   │   └── chemokine_assay.py  # Chemotactic response pipeline
│   │
│   ├── preprocessing/          # Data preparation
│   │   ├── prep_clickpoints_databases.py
│   │   ├── drift_correction.py
│   │   └── prep_6layer_tiffs_for_detection_ben.py
│   │
│   ├── tracking/               # Cell detection and tracking
│   │   ├── cell_tracker.py     # Main tracking entry point
│   │   ├── tracking_functions.py
│   │   ├── NK_cell_weights.h5  # Pre-trained U-Net weights
│   │   └── unet/               # U-Net model architecture
│   │
│   ├── postprocessing/         # Analysis and filtering
│   │   ├── motility_filter_cdb.py
│   │   └── write_to_excel.py
│   │
│   ├── plots/                  # Visualization
│   │   ├── plot_mf_speed_pers.py
│   │   ├── plot_kde_speed_pers.py
│   │   ├── plot_chemokine_assay.py
│   │   └── plot_quadrants_stacked.py
│   │
│   ├── pooling/                # Aggregate analysis
│   │   ├── pool_kde_plots.py
│   │   └── pool_mf_speed_pers.py
│   │
│   └── utils.py                # Utility functions
│
├── setup.py
├── LICENSE
└── README.md
```

## Configuration

### Tracking Parameters

Key parameters can be adjusted in `tracking/tracking_functions.py`:
- `minimal_cell_size`: Minimum pixels for valid cell detection (default: 30)
- `z_lim`: Z-limit for cell overlap calculation
- `max_cost`: Maximum cost for track stitching (default: 50)

### U-Net Configuration

Model parameters in `tracking/unet/unet_config.py`:
- Input: 12 channels (3D image stacks with temporal context)
- Output: 2 classes (cell/background)
- Augmentation options for retraining

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite the repository.
