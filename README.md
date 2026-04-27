# geotempfill

**Tensor-completion gap filling for NOAA weather station data.**

`geotempfill` is a Python project for reconstructing missing temperature observations in multi-station weather datasets using **High-accuracy Low-Rank Tensor Completion (HaLRTC)**.

The project downloads real-world climate observations from NOAA's **GHCN-Daily** archive, converts the data into a `(variable × time × station)` tensor, hides a subset of observed values for evaluation, fills the missing entries using HaLRTC, and compares the result against classical baseline methods.

This project was developed for the **Geospatial Processing** course project, AA 2025–2026, at **Politecnico di Milano**.

---

## 1. Project Motivation

Weather station datasets are often incomplete. Missing observations may be caused by station outages, sensor failures, short reporting periods, irregular station coverage, or gaps in historical records.

Traditional interpolation methods usually focus on only one type of structure:

- spatial interpolation uses nearby stations;
- temporal interpolation uses previous or later observations;
- simple averaging ignores both detailed spatial and temporal patterns.

However, weather station data naturally has a multi-dimensional structure:

1. **Temporal correlation**  
   Temperatures are continuous over time and often show seasonal patterns.

2. **Spatial correlation**  
   Nearby stations usually report similar weather conditions.

3. **Cross-variable correlation**  
   Variables such as `TMAX` and `TMIN` are strongly related.

This project uses tensor completion because it can exploit these dimensions simultaneously.

---

## 2. Methodological Background

The methodology is inspired by:

> Liao, Y. et al. (2024).  
> *A new disease mapping method for improving data completeness of syndromic surveillance with high missing rates.*  
> **Transactions in GIS**, 28(6), 1869–1882.  
> https://doi.org/10.1111/tgis.13200

The original paper applied HaLRTC to a `(virus × quarter × city)` tensor for syndromic surveillance data.  
This project transfers the same idea to geospatial weather observations, using a:

```text
(variable × time × station)
```

tensor.

---

## 3. Dataset

### 3.1 Data Source

The project uses the **NOAA NCEI GHCN-Daily** dataset:

- GHCN-Daily = Global Historical Climatology Network Daily
- Source: NOAA National Centers for Environmental Information
- Access: public HTTPS endpoints
- API key: not required

The downloader retrieves station metadata and per-station daily CSV files from NOAA.

### 3.2 Study Area

The selected study area is:

```text
California, USA
```

California is suitable because:

- it has many NOAA GHCN stations;
- it has strong geographic diversity;
- it includes coastal, inland, desert, mountain, and valley regions;
- temperature gradients are spatially meaningful;
- the state is large enough to show spatial variation but still manageable for a one-person project.

### 3.3 Demo Configuration

The default demo uses:

```text
Region: California (CA)
Years: 2020–2021
Variables: TMAX, TMIN
Stations: 30
Temporal aggregation: monthly
```

The resulting tensor shape is:

```text
(variables, time, stations) = (2, 24, 30)
```

where:

- `2` = `TMAX`, `TMIN`
- `24` = 24 monthly time steps from 2020 to 2021
- `30` = selected NOAA stations with valid temperature observations

---

## 4. Project Pipeline

The full workflow is:

```text
NOAA data download
        ↓
Station filtering
        ↓
Daily-to-monthly aggregation
        ↓
Tensor construction
        ↓
Random hold-out masking
        ↓
HaLRTC tensor completion
        ↓
Baseline comparison
        ↓
Metric evaluation
        ↓
Figure and report generation
```

---

## 5. Tensor Construction

The project constructs a third-order tensor:

```text
X ∈ R^(V × T × S)
```

where:

- `V` = weather variables
- `T` = time steps
- `S` = weather stations

For the default California demo:

```text
V = 2
T = 24
S = 30
```

The tensor is accompanied by a boolean mask:

```text
mask[v, t, s] = True   if the value is observed
mask[v, t, s] = False  if the value is missing
```

This mask is required by HaLRTC because observed values must remain fixed during completion.

---

## 6. HaLRTC Algorithm

### 6.1 Problem Definition

The goal is to reconstruct a complete tensor `X` from a partially observed tensor `M`.

The observed entries are indexed by `Ω`.

The constraint is:

```text
X_Ω = M_Ω
```

This means that the completed tensor must preserve all known observations.

### 6.2 Low-Rank Tensor Completion

Direct tensor rank minimisation is difficult, so HaLRTC uses a convex relaxation based on the nuclear norm of tensor unfoldings.

The optimization problem is:

```text
minimize    Σ_n α_n ||X_(n)||_*
subject to  X_Ω = M_Ω
```

where:

- `X_(n)` is the mode-`n` unfolding of the tensor;
- `|| · ||_*` is the matrix nuclear norm;
- `α_n` controls the contribution of each tensor mode.

### 6.3 ADMM Solver

The implementation follows an ADMM-style iterative procedure:

1. Unfold the tensor along each mode.
2. Apply Singular Value Thresholding (SVT).
3. Fold matrices back into tensors.
4. Average auxiliary tensors.
5. Re-impose observed values.
6. Update dual variables.
7. Stop when relative change is below tolerance.

The implementation is written in NumPy and does not rely on deep learning frameworks.

---

## 7. Baseline Methods

The project compares HaLRTC with three simpler baselines.

### 7.1 MeanFill

Fills missing values using the mean value for each `(variable, station)` pair.

This captures station-level climatology but ignores time variation.

### 7.2 TemporalMean

Fills missing values using the mean of all observed stations at the same `(variable, time)` slice.

This captures temporal variation but ignores station identity.

### 7.3 IDW

Uses Inverse Distance Weighting over station coordinates.

For each missing value, nearby stations at the same time step contribute more strongly than distant stations.

---

## 8. Evaluation Metrics

The project evaluates predictions on held-out observed entries.

Metrics include:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: coefficient of determination
- **Pearson r**: linear correlation between predicted and true values

The demo hides 10% of originally observed entries and evaluates each method only on those hidden entries.

---

## 9. Demo Results

A successful California demo produced the following result:

```text
Method                RMSE        MAE        R^2          r        n
----------------------------------------------------------------------
HaLRTC              1.9420     1.5073     0.9620     0.9828      135
MeanFill            6.6937     5.5346     0.5483     0.7419      135
TemporalMean        4.3507     3.3805     0.8092     0.9005      135
IDW                 3.6755     2.6431     0.8638     0.9334      135
```

### Interpretation

HaLRTC achieves the best performance:

- lowest RMSE;
- lowest MAE;
- highest R²;
- highest Pearson correlation.

This indicates that tensor completion successfully exploits joint spatial, temporal, and variable-level structure.

---

## 10. Repository Structure

```text
geotempfill/
├── src/
│   └── geotempfill/
│       ├── __init__.py
│       ├── __main__.py
│       ├── baselines.py
│       ├── cli.py
│       ├── data.py
│       ├── evaluation.py
│       ├── halrtc.py
│       ├── tensor.py
│       ├── visualize.py
│       └── py.typed
│
├── examples/
│   └── run_california_demo.py
│
├── tests/
│   ├── test_baselines.py
│   ├── test_evaluation.py
│   ├── test_halrtc.py
│   └── test_tensor.py
│
├── data/
│   ├── raw/
│   ├── cache/
│   ├── processed/
│   └── sample/
│
├── results/
│   ├── figures/
│   └── reports/
│
├── .gitignore
├── environment.yml
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## 11. Installation

### 11.1 Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 11.2 Install the Package

From the project root:

```powershell
pip install -e ".[dev]"
```

### 11.3 Alternative: Conda

```bash
conda env create -f environment.yml
conda activate geotempfill
```

---

## 12. Running the Project

### 12.1 Run the Full California Demo

From the project root:

```powershell
python .\examples\run_california_demo.py
```

This will:

1. download NOAA GHCN-Daily data;
2. select stations with valid temperature observations;
3. build a monthly tensor;
4. hide 10% of observed entries;
5. run HaLRTC;
6. run baseline methods;
7. compute evaluation metrics;
8. save figures and JSON report.

### 12.2 Expected Output

The script prints progress similar to:

```text
[1/6] Downloading NOAA GHCN-Daily data...
[2/6] Building monthly tensor...
[3/6] Creating held-out test entries...
[4/6] Running HaLRTC...
[5/6] Running baselines...
[6/6] Saving figures...
Demo completed successfully.
```

---

## 13. Command-Line Interface

The project also provides a CLI.

### 13.1 Show Help

```powershell
python -m geotempfill --help
```

### 13.2 Download Data

```powershell
python -m geotempfill download `
  --state CA `
  --start-year 2020 `
  --end-year 2021 `
  --variables TMAX TMIN `
  --max-stations 30 `
  --out-dir data/raw `
  --cache-dir data/cache
```

### 13.3 Run Benchmark

```powershell
python -m geotempfill benchmark `
  --obs data/raw/observations_CA.csv `
  --stations data/raw/stations_CA.csv `
  --variables TMAX TMIN `
  --freq MS `
  --hide-fraction 0.1 `
  --rho 0.005 `
  --max-iter 300 `
  --report results/reports/benchmark_CA.json
```

---

## 14. Python API Example

```python
import geotempfill as gtf
import numpy as np

# Download California data
obs, stations = gtf.fetch_state_data(
    "CA",
    variables=["TMAX", "TMIN"],
    years=[2020, 2021],
    max_stations=30,
    cache_dir="data/cache",
)

# Build tensor
tensor = gtf.build_tensor(
    obs,
    variables=["TMAX", "TMIN"],
    time_col="date",
    station_col="station",
    freq="MS",
    station_coords=stations,
)

# Prepare data for algorithms
data = tensor.fill_for_algorithm()

# Hide 10% of observed entries
rng = np.random.default_rng(0)
train_mask, holdout = gtf.hide_random(tensor.mask, 0.1, rng=rng)

# Run HaLRTC
result = gtf.halrtc(
    data,
    train_mask,
    rho=5e-3,
    max_iter=300,
    tol=1e-5,
)

# Evaluate
metrics = gtf.score(data, result.completed, holdout)

print(metrics)
```

---

## 15. Outputs

After running the demo, the project creates:

### 15.1 Raw Data

```text
data/raw/observations_CA.csv
data/raw/stations_CA.csv
```

### 15.2 Reports

```text
results/reports/california_demo_metrics.json
```

### 15.3 Figures

```text
results/figures/california_station_map.png
results/figures/california_missing_heatmap.png
results/figures/california_halrtc_convergence.png
results/figures/california_method_comparison.png
```

---

## 16. Testing

Run all tests:

```powershell
pytest
```

Run a specific test file:

```powershell
pytest tests/test_halrtc.py -v
```

The test suite covers:

- tensor construction;
- HaLRTC utilities;
- singular value thresholding;
- baseline methods;
- evaluation metrics;
- masking logic.

---

## 17. Main Modules

### `data.py`

Downloads NOAA station metadata and observations.

Main functions:

- `list_stations`
- `fetch_station_data`
- `fetch_state_data`

### `tensor.py`

Builds a dense tensor and observation mask.

Main function:

- `build_tensor`

Main class:

- `WeatherTensor`

### `halrtc.py`

Implements HaLRTC.

Main functions:

- `halrtc`
- `unfold`
- `fold`
- `svt`

Main class:

- `HaLRTCResult`

### `baselines.py`

Implements baseline methods.

Main functions:

- `mean_fill`
- `temporal_mean_fill`
- `idw_fill`

### `evaluation.py`

Implements hold-out masking and metrics.

Main functions:

- `hide_random`
- `score`

Main class:

- `Metrics`

### `visualize.py`

Implements plotting helpers.

Main functions:

- `plot_station_map`
- `plot_missing_heatmap`
- `plot_convergence`
- `plot_method_comparison`

---

## 18. Notes on Station Selection

The NOAA GHCN-Daily archive contains many different station types. Some stations, especially volunteer stations with IDs beginning with `US1`, often report precipitation only and do not contain temperature variables such as `TMAX` or `TMIN`.

To avoid downloading many precipitation-only stations, the project prioritizes stations with IDs such as:

- `USW`
- `USC`
- `USR`

and keeps only stations that contain at least one non-missing requested variable in the selected year range.

---

## 19. Limitations

This project is a simplified course-project implementation.

Current limitations include:

- only a NumPy implementation of HaLRTC;
- no GPU acceleration;
- no advanced hyperparameter tuning;
- no uncertainty quantification;
- no direct comparison with full kriging or Gaussian Process models;
- the default experiment uses a limited number of stations for reproducibility.

---

## 20. Future Work

Possible extensions include:

- larger-scale experiments across multiple states;
- adding more variables such as precipitation, wind, and humidity;
- using seasonal decomposition before tensor completion;
- comparing with kriging or Gaussian Process regression;
- implementing a PyTorch or GPU-accelerated version;
- adding uncertainty estimates for reconstructed values;
- using spatial clustering before tensor completion.

---

## 21. License

This project is released under the MIT License. See `LICENSE` for details.

---

## 22. Citation

If using this project in academic work, please cite the methodological reference:

```bibtex
@article{Liao2024HaLRTC,
  title={A new disease mapping method for improving data completeness of syndromic surveillance with high missing rates},
  author={Liao, Yilan and Shi, Yuanhao and Fan, Zhirui and Zhu, Zhiyu and Huang, Binghu and Du, Wei and Wang, Jinfeng and Wang, Liping},
  journal={Transactions in GIS},
  volume={28},
  number={6},
  pages={1869--1882},
  year={2024},
  publisher={Wiley},
  doi={10.1111/tgis.13200}
}
```

---

## 23. Summary

`geotempfill` demonstrates that low-rank tensor completion is a useful framework for reconstructing missing geospatial temperature observations.

In the California experiment, HaLRTC clearly outperforms simple mean filling, temporal averaging, and inverse-distance weighting, showing the value of modelling weather data as a structured tensor.
