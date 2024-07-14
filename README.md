### MetricX CUDA

An attempt at translating [google-research/metricx](https://github.com/google-research/metricx) to low-level CUDA.


### Setup
```sh
# pip install pytest pytest-cov
python -m pytest --cov=. --cov-report xml:cov.xml test.py
```