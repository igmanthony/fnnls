# fnnls
Rust port of Python port of MATLAB `FNNLSa` algorithm

[Original MATLAB by Rasmus Bro](http://www.mathworks.com/matlabcentral/fileexchange/3388-nnls-and-constrained-regression?focused=5051382&tab=function)

[Python port by Daniel Elnatan](https://github.com/delnatan/FNNLSa)

`fnnls` requires the `ndarray` and `ndarray-linalg` crates.
As far as I know, `fnnls` provides ~~the same~~ corrected output from the Python port above.

A, poorly-formatted, updated `fnnls.py` file is provided in the main directory in case it is useful. I used this `fnnls.py` file as well as the original Rasmus Bro MATLAB file to try to fix an indexing error I believe was introduced in the source `fnnls.py` file.

### Benchmarks have not been updated after indexing bug fix but should be similar to below
Benchmarks for the same [600 x 1000] transformation matrix:
- Python `scipy.optimize.nnls` timeit: `35.23 ms per loop; n = 1000`
- Python `FNNLSa` timeit: `5.30 ms per loop; n = 1000`
- Rust `fnnls` criterion: `1.62 ms per loop; n = 5050`

