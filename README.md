# geotempfill

**Tensor-completion missing-value filling for multidimensional data, with a weather-station reconstruction workflow.**

GeoTempFill is a lightweight Python toolkit for filling missing values in sparse multidimensional data represented as tensors. The core design is a reusable pipeline that converts long-format observations into a tensor plus observation mask, applies a selected completion method, and evaluates the completed result on held-out observed entries.

The main implemented workflow is weather station missing-value reconstruction. In this project, the toolkit is developed and evaluated against the structure of NOAA weather-station data, where observations vary across variables, time, and stations. The same tensor-completion design can also be applied to other multidimensional missing-data settings.

The toolkit is useful when missing values are structured across multiple dimensions, for example:

- `variable x time x station` weather observations, which are the main workflow in this repository;
- `syndrome x quarter x city` surveillance data;
- `virus x week x city x age_group` public-health or environmental panels;
- other long-format datasets that can be represented as an N-dimensional tensor.

---

## Core Features

- Build dense tensors and observation masks from long-format data.
- Support weather-specific `variable x time x station` tensors.
- Support generic N-D tensors for other multidimensional missing-data problems.
- Run HaLRTC tensor completion with a NumPy implementation.
- Add optional location-aware smoothing using station coordinates.
- Apply elevation-based physical correction for temperature variables.
- Compare against mean fill, temporal mean, IDW, ordinary kriging, simple cokriging, and empirical-Bayes baselines.
- Evaluate predictions on randomly held-out observed entries.
- Generate reports and figures for the California demo workflow.

---

## Performance Snapshot

The current California demo uses:

```text
Region: California (CA)
Years: 2020-2021
Variables: TMAX, TMIN, ADPT, ASLP, AWBT
Stations: 30
Temporal aggregation: monthly
Tensor shape: (5, 24, 30)
Holdout: 10% of originally observed entries
```

A successful run produced the following per-variable results for the physically corrected, standardized, location-aware HaLRTC workflow:

```text
PhysicalLocationHaLRTC
Variable         RMSE        MAE        R^2          r        n
------------------------------------------------------------
TMAX           0.8184     0.6499     0.9888     0.9960       67
TMIN           0.7680     0.6161     0.9837     0.9931       65
ADPT           9.7422     6.7650     0.9735     0.9924       72
ASLP           6.0405     3.6234     0.9799     0.9938       71
AWBT           5.2712     3.6122     0.9863     0.9956       85
```

In this experiment, HaLRTC is the strongest overall method across the five variables. MeanFill, TemporalMean, and IDW show larger RMSE values across all variables, while Cokriging is a competitive multivariable spatial baseline on some variables such as AWBT.

The main evaluation is reported per variable because the variables have different physical units and scales.

---

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Install the package from the project root:

```powershell
pip install -e ".[dev]"
```

Alternative Conda setup:

```bash
conda env create -f environment.yml
conda activate geotempfill
```

---

## Quick Start: Weather Tensor Completion

```python
import geotempfill as gtf
import numpy as np

variables = ["TMAX", "TMIN", "ADPT", "ASLP", "AWBT"]

obs, stations = gtf.fetch_state_data(
    "CA",
    variables=variables,
    years=[2020, 2021],
    max_stations=30,
    cache_dir="data/cache",
)

tensor = gtf.build_tensor(
    obs,
    variables=variables,
    time_col="date",
    station_col="station",
    freq="MS",
    station_coords=stations,
)

data = tensor.fill_for_algorithm()
rng = np.random.default_rng(0)
train_mask, holdout = gtf.hide_random(tensor.mask, 0.1, rng=rng)

coords = tensor.station_coords[["latitude", "longitude"]].to_numpy()

result = gtf.halrtc(
    data,
    train_mask,
    rho=5e-3,
    max_iter=300,
    tol=1e-5,
    coords=coords,
    spatial_weight=0.10,
    spatial_power=2.0,
    station_mode=2,
)

completed = np.where(train_mask, data, result.completed)
metrics = gtf.score(data, completed, holdout)

print(metrics)
```

The full California demo additionally applies variable-wise standardization and elevation-based physical correction before HaLRTC, then transforms predictions back to the original physical units.

---

## Advanced Usage: User-Defined Tensor Axes

Although the main workflow is built around weather station reconstruction, the tensor construction layer can also be used when the tensor axes are user-defined columns rather than the weather-specific `variable x time x station` structure.

Use `build_nd_tensor` to construct a tensor from long-format data with arbitrary axis names:

```python
import geotempfill as gtf

tensor = gtf.build_nd_tensor(
    df,
    index_cols=["syndrome", "quarter", "city"],
    value_col="count",
)
```

This can still be a 3-D tensor, such as `syndrome x quarter x city`. The difference is that the axes are not tied to weather variables, monthly time steps, or station metadata. It also supports higher-dimensional layouts such as `virus x week x city x age_group`.

---

## California Demo

Run the full demo from the project root:

```powershell
python .\examples\run_california_demo.py
```

This will:

1. download NOAA GHCN-Daily data or load cached CSV files;
2. select stations with valid requested observations;
3. build a monthly `variable x time x station` tensor;
4. hide 10% of observed entries for evaluation;
5. run physically corrected, standardized, location-aware HaLRTC;
6. run baseline methods;
7. compute per-variable evaluation metrics;
8. save figures and a JSON report.

The default run writes outputs under:

```text
results/reports/missing_10pct_seed0/
results/figures/missing_10pct_seed0/
```

You can change the holdout rate, seed, station count, spatial smoothing weight, and optional baselines:

```powershell
python .\examples\run_california_demo.py `
  --hide-fraction 0.3 `
  --seed 0 `
  --max-stations 30 `
  --spatial-weight 0.10
```

---

## Command-Line Interface

Show help:

```powershell
python -m geotempfill --help
```

Download NOAA GHCN-Daily data:

```powershell
python -m geotempfill download `
  --state CA `
  --start-year 2020 `
  --end-year 2021 `
  --variables TMAX TMIN ADPT ASLP AWBT `
  --max-stations 30 `
  --out-dir data/raw `
  --cache-dir data/cache
```

Run a benchmark from downloaded CSV files:

```powershell
python -m geotempfill benchmark `
  --obs data/raw/observations_CA.csv `
  --stations data/raw/stations_CA.csv `
  --variables TMAX TMIN ADPT ASLP AWBT `
  --freq MS `
  --hide-fraction 0.1 `
  --seed 0 `
  --methods halrtc mean temporal idw kriging cokriging empirical_bayes `
  --rho 0.005 `
  --max-iter 300 `
  --report results/reports/benchmark_CA.json
```

Use `--methods` to choose which completion methods to run. Available method names include `halrtc`, `mean`, `temporal`, `idw`, `kriging`, `cokriging`, and `empirical_bayes`.

The California demo uses 2020-2021 and a 10% holdout by default. The lower-level CLI has its own defaults, so explicit arguments are recommended for reproducible runs.

---

## Method Summary: HaLRTC

HaLRTC is the core completion method in GeoTempFill. It is designed for tensors whose missing values are not independent, but are structured across several related axes. In the weather-station workflow, those axes are variables, time steps, and stations.

The method is inspired by:

> Liao, Y. et al. (2024). *A new disease mapping method for improving data completeness of syndromic surveillance with high missing rates.* Transactions in GIS, 28(6), 1869-1882. https://doi.org/10.1111/tgis.13200

The original paper applied HaLRTC to a `virus x quarter x city` tensor for syndromic surveillance data. This project transfers the same idea to geospatial weather observations, using a `variable x time x station` tensor.

Weather station data naturally has a multi-dimensional structure:

- temporal correlation: temperatures are continuous over time and often show seasonal patterns;
- spatial correlation: nearby stations usually report similar weather conditions;
- cross-variable correlation: variables such as `TMAX` and `TMIN` are strongly related.

HaLRTC reconstructs a complete tensor from a partially observed tensor while preserving the known observations. Instead of filling each variable, station, or time series separately, it searches for a low-rank structure shared across tensor modes. This is the main reason it is a natural fit for multidimensional missing-value completion.

Direct tensor rank minimization is difficult, so HaLRTC uses a convex relaxation based on the nuclear norm of tensor unfoldings. In practice, the implementation uses an ADMM-style iterative procedure:

1. unfold the tensor along each mode;
2. apply singular value thresholding;
3. fold matrices back into tensors;
4. average auxiliary tensors;
5. re-impose observed values;
6. stop when relative change is below tolerance.

The implementation is written in NumPy and does not rely on deep learning frameworks.

The current demo extends the base HaLRTC workflow with weather-aware preprocessing and spatial information:

- inverse-distance station smoothing during completion;
- elevation-based lapse-rate correction for `TMAX` and `TMIN`;
- variable-wise standardization using training entries only.

---

## Baseline Methods

The project compares HaLRTC with six simpler baselines:

- **MeanFill**: fills missing values using the mean value for each `(variable, station)` pair.
- **TemporalMean**: fills missing values using the mean of all observed stations at the same `(variable, time)` slice.
- **IDW**: uses inverse distance weighting over station coordinates.
- **Ordinary Kriging**: uses an exponential covariance model over station coordinates and falls back to IDW when too few stations are observed.
- **Cokriging (simple)**: uses a lightweight separable covariance model based on variable correlation and station distance.
- **Empirical Bayes**: fits a variable-specific additive decomposition with time and station effects on observed entries only.

---

## Outputs

After running the demo, the project creates:

```text
data/raw/observations_CA.csv
data/raw/stations_CA.csv
results/reports/missing_<X>pct_seed<S>/california_demo_metrics.json
results/figures/missing_<X>pct_seed<S>/california_station_map.png
results/figures/missing_<X>pct_seed<S>/california_missing_heatmap.png
results/figures/missing_<X>pct_seed<S>/california_halrtc_convergence.png
results/figures/missing_<X>pct_seed<S>/california_station_error_map.png
```

where `<X>` comes from `--hide-fraction` and `<S>` from `--seed`.

---

## Main Modules

```text
src/geotempfill/
  data.py        NOAA station metadata and observations
  tensor.py      weather-specific 3-D tensors and generic N-D tensors
  halrtc.py      HaLRTC, tensor unfolding utilities, physical correction
  baselines.py   mean fill, temporal mean, IDW
  spatial.py     ordinary kriging and simple cokriging
  bayesian.py    empirical-Bayes additive baseline
  methods.py     method registry used by the CLI and demo
  evaluation.py  hold-out masking and metrics
  visualize.py   plotting helpers
  cli.py         command-line interface
```

The main public API is available through:

```python
import geotempfill as gtf
```

---

## Testing

Run all tests:

```powershell
pytest
```

Run a specific test file:

```powershell
pytest tests/test_halrtc.py -v
```

The test suite covers tensor construction, HaLRTC utilities, singular value thresholding, baseline methods, evaluation metrics, masking logic, and CLI smoke tests.

---

## Limitations

This project is a simplified course-project implementation.

Current limitations include:

- only a NumPy implementation of HaLRTC;
- no GPU acceleration;
- no advanced hyperparameter tuning;
- no uncertainty quantification;
- the default experiment uses a limited number of stations for reproducibility;
- physical correction is currently applied only to temperature variables;
- pressure correction is not yet physically modelled with a logarithmic pressure-height relationship.

Possible extensions include larger-scale experiments across multiple states, additional weather variables, seasonal decomposition, GPU acceleration, uncertainty estimates, spatial clustering, and systematic tuning of the spatial smoothing weight.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Citation

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

## Summary

`geotempfill` demonstrates that low-rank tensor completion is a useful framework for reconstructing missing geospatial weather observations. The project shows the value of modelling weather station data as a structured tensor while also incorporating geographic distance and elevation-based physical correction.
