# ds-anomaly-detection-lab

A compact anomaly detection toolkit for learning statistical outlier detection on one-dimensional time-series or metric data.

## Supported Methods

- Z-score detection for normally distributed numeric signals.
- Interquartile range detection for simple robust outlier bounds.
- Median absolute deviation detection for data with strong outliers.
- Isolation Forest through `AdvancedAnomalyDetector` when scikit-learn is installed.

`DetectionMethod.DBSCAN` is currently listed in code but not implemented in the statistical detector. Track that before presenting DBSCAN as a supported method in tutorials.

## Quick Start

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install numpy pytest
python -m pytest
```

## Usage

```python
from src.anomaly_detector import DetectionMethod, StatisticalAnomalyDetector

data = [48, 52, 49, 51, 200, 50, 48, 52, 49]
detector = StatisticalAnomalyDetector(
    method=DetectionMethod.IQR,
    threshold=1.5,
)

result = detector.detect(data)
print(result.indices)
print(result.summary())
```

## Development

Run the test suite before opening a pull request:

```bash
python -m pytest
```

The tests cover statistical detection methods, empty input, constant data, threshold behavior, and NumPy array input.

## Limitations

- Statistical methods assume finite numeric data.
- The current detector focuses on one-dimensional signals.
- Isolation Forest requires scikit-learn and is optional.
- There is not yet a standardized package configuration file.
