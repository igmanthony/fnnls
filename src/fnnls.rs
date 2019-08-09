#![allow(non_snake_case)]

use ndarray::{Dimension, prelude::*};
use ndarray_linalg::Solve;
use std::f64::EPSILON;


pub fn fnnls(xtx: &Array2<f64>, xty: &Array1<f64>)
    -> (Array1<f64>, Array1<f64>) {
    let (M, N) = (xtx.rows(), xtx.cols());
    let mut P = Array::zeros(M);               // passive; indices with vals > 0
    let mut Z = Array::from_iter(0..N) + 1;            // active; i w/ vals <= 0
    let mut ZZ = Array::from_iter(0..N) + 1;               // working active set
    let mut x: Array1<f64> = Array::zeros(N);         // initial solution vector
    let mut w = xty - &(xtx.dot(&x));                           // weight vector
    let mut it = 0;                               // iterator for the while loop
    let itmax = 30 * N;                                    // maximum iterations

    // Continue if indices in the active set or values > than machine epsilon
    while Z.iter().any(|&i| i > 0) && ZZ.iter().any(|&i| &w[i - 1] > &EPSILON)
    {
        let t = max_index(&(ZZ.mapv(|i| w[i - 1]))) + 1;
        P[ZZ[t - 1] - 1] = ZZ[t - 1];                 // move to the passive set
        Z[ZZ[t - 1] - 1] = 0;                      // remove from the active set
        ZZ = Array::from_vec(find_nonzero(&Z)) + 1;
        let mut PP = Array::from_vec(find_nonzero(&P)) + 1;
        let mut PPcopy = Array::from_vec(find_nonzero(&P)) + 1;
        let mut s: Array1<f64> = Array::zeros(N);              // trial solution
        match PP.len() {
            0 => s[0] = 0.0,
            1 => s[PP[0] - 1] = xty[PP[0] - 1] / xtx[[PP[0] - 1, PP[0] - 1]],
            _ => {
                let xtx_pp_solution = slice_with_array(xtx, &(PPcopy - 1))
                    .solve_into(PP.mapv(|i| xty[i - 1]))
                    .unwrap();              // solve PP-reduced set of xtx @ xty
                for (i, &value) in PP.indexed_iter() {
                    s[value - 1] = xtx_pp_solution[i];
                }
            }
        }
        for &i in ZZ.iter() {
            s[i - 1] = 0.0;                      // set active coefficients to 0
        }

        while (&PP).iter().any(|&i| &s[i - 1] <= &EPSILON) && it < itmax {
            it += 1;
            let s_mask = s.mapv(|e| e <= EPSILON);
            let tmp = P
                .indexed_iter()
                .map(|(i, &v)| if s_mask[[i]] { v } else { 0 })
                .collect::<Vec<usize>>();
            let QQ = Array::from_vec(find_nonzero_vec(&tmp)) + 1;
            let xQQ = QQ.mapv(|i| x[i - 1]);
            let alpha = min(&(&xQQ / &(&xQQ - &QQ.mapv(|i| s[i - 1]))));
            x = &x + &(alpha * (&s - &x));
            let mask = P.mapv(|i| i != 0) & x.mapv(|i| i.abs() < EPSILON);
            for (i, &v) in mask.indexed_iter() {
                if v {
                    Z[i] = i + 1;
                    P[i] = 0;
                }
            }
            PP = Array::from_vec(find_nonzero(&P)) + 1;
            PPcopy = Array::from_vec(find_nonzero(&P)) + 1;
            ZZ = Array::from_vec(find_nonzero(&Z)) + 1;
            match PP.len() {
                // verbatim repeat of the previous match statement
                0 => s[0] = 0.0,
                1 => {
                    s[PP[0] - 1] = xty[PP[0] - 1] / xtx[[PP[0] - 1, PP[0] - 1]]
                }
                _ => {
                    let xtx_pp_solution = slice_with_array(xtx, &(PPcopy - 1))
                        .solve_into(PP.mapv(|i| xty[i - 1]))
                        .unwrap();
                    for (i, &value) in PP.indexed_iter() {
                        s[value - 1] = xtx_pp_solution[i];
                    }
                }
            }
            for &i in ZZ.iter() {
                s[i - 1] = 0.0;
            }
        }
        x = s;                               // assign current solution (s) to x
        w = xty - &(xtx.dot(&x));                           // recompute weights
    }
    (x, w)
}


pub fn find_nonzero_vec(vec: &Vec<usize>) -> Vec<usize> {
    vec.iter()
       .enumerate()
       .filter(|(_, &value)| value != 0)
       .map(|(i, _)| i)
       .collect::<Vec<usize>>()
}


pub fn find_nonzero(array: &Array1<usize>) -> Vec<usize> {
    array.indexed_iter()
         .filter(|(_, &value)| value !=0)
         .map(|(i, _)| i)
         .collect::<Vec<usize>>()
}


pub fn slice_with_array(a: &Array2<f64>, b: &Array1<usize>) ->
                        Array2<f64> {
    let mut newArray = Array::zeros((b.len(), b.len()));
    for i in 0..b.len() {
        for j in 0..b.len() {
            newArray[[i, j]] = a[[b[j], b[i]]]
        }
    }
    newArray
}


pub fn max_index(array: &Array1<f64>) -> usize {
    let mut index = 0;
    for (i, &value) in array.iter().enumerate() {
        if value > array[index] {
            index = i;
        }
    }
    index
}


pub fn min<D: Dimension>(array: &Array<f64, D>) -> f64 {
    array.iter().cloned().fold(1.0/0.0, f64::min)
}