### MetricX CUDA

An attempt at translating [google-research/metricx](https://github.com/google-research/metricx) to low-level CUDA.


### Setup
```sh
# pip install pytest pytest-cov
python -m pytest --cov=. --cov-report xml:cov.xml test.py

# pip install flameprof
flameprof profile.prof --font-size 8 --width 2000 > profile.svg

# pip install gprof2dot && brew install graphviz
gprof2dot -f pstats profile.prof | dot -Tpng -o profile.png
```