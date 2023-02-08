use fnnls::fnnls;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "pyfnnls")]
fn python(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Formats the sum of two numbers as string.
    #[pyfn(m)]
    #[pyo3(name = "fnnls")]
    fn fnnls_wrapper<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'_, f64>,
        b: PyReadonlyArray1<'_, f64>,
    ) -> &'py PyArray1<f64> {
        let (res, _w) = fnnls(&a.as_array().to_owned(), &b.as_array().to_owned());

        return res.into_pyarray(py);
    }
    Ok(())
}
