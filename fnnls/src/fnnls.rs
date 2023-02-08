#![allow(non_snake_case)]

use ndarray::{prelude::*, Dimension};
use ndarray_linalg::Solve;
use std::f64::EPSILON;

pub fn fnnls(xtx: &Array2<f64>, xty: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let (M, N) = (xtx.nrows(), xtx.ncols());
    let mut P = Array::zeros(M); // passive; indices with vals > 0
    let mut Z = Array::from_iter(0..N) + 1; // active; i w/ vals <= 0
    let mut ZZ = Array::from_iter(0..N) + 1; // working active set
    let mut x: Array1<f64> = Array::zeros(N); // initial solution vector
    let mut w = xty - &(xtx.dot(&x)); // weight vector
    let mut it = 0; // iterator for the while loop
    let itmax = 30 * N; // maximum iterations

    // Continue if indices in the active set or values > than machine epsilon
    while Z.iter().any(|&i| i > 0) && ZZ.iter().any(|&i| w[i - 1] > EPSILON) {
        let t = max_index(&(ZZ.mapv(|i| w[i - 1]))) + 1;
        P[ZZ[t - 1] - 1] = ZZ[t - 1]; // move to the passive set
        Z[ZZ[t - 1] - 1] = 0; // remove from the active set
        ZZ = Array::from(find_nonzero(&Z)) + 1;
        let mut PP = Array::from(find_nonzero(&P)) + 1;
        let mut PPcopy = Array::from(find_nonzero(&P)) + 1;
        let mut s: Array1<f64> = Array::zeros(N); // trial solution
        match PP.len() {
            0 => s[0] = 0.0,
            1 => s[PP[0] - 1] = xty[PP[0] - 1] / xtx[[PP[0] - 1, PP[0] - 1]],
            _ => {
                let xtx_pp_solution = slice_with_array(xtx, &(PPcopy - 1))
                    .solve_into(PP.mapv(|i| xty[i - 1]))
                    .unwrap(); // solve PP-reduced set of xtx @ xty
                for (i, &value) in PP.indexed_iter() {
                    s[value - 1] = xtx_pp_solution[i];
                }
            }
        }
        for &i in ZZ.iter() {
            s[i - 1] = 0.0; // set active coefficients to 0
        }

        while PP.iter().any(|&i| s[i - 1] <= EPSILON) && it < itmax {
            it += 1;
            let s_mask = s.mapv(|e| e <= EPSILON);
            let tmp = P
                .indexed_iter()
                .map(|(i, &v)| if s_mask[[i]] { v } else { 0 })
                .collect::<Vec<usize>>();
            let QQ = Array::from(find_nonzero_vec(&tmp)) + 1;
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
            PP = Array::from(find_nonzero(&P)) + 1;
            PPcopy = Array::from(find_nonzero(&P)) + 1;
            ZZ = Array::from(find_nonzero(&Z)) + 1;
            match PP.len() {
                // verbatim repeat of the previous match statement
                0 => s[0] = 0.0,
                1 => s[PP[0] - 1] = xty[PP[0] - 1] / xtx[[PP[0] - 1, PP[0] - 1]],
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
        x = s; // assign current solution (s) to x
        w = xty - &(xtx.dot(&x)); // recompute weights
    }
    (x, w)
}

pub fn find_nonzero_vec(vec: &[usize]) -> Vec<usize> {
    vec.iter()
        .enumerate()
        .filter(|(_, &value)| value != 0)
        .map(|(i, _)| i)
        .collect::<Vec<usize>>()
}

pub fn find_nonzero(array: &Array1<usize>) -> Vec<usize> {
    array
        .indexed_iter()
        .filter(|(_, &value)| value != 0)
        .map(|(i, _)| i)
        .collect::<Vec<usize>>()
}

pub fn slice_with_array(a: &Array2<f64>, b: &Array1<usize>) -> Array2<f64> {
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
    array.iter().cloned().fold(1.0 / 0.0, f64::min)
}

pub fn test_setup() -> (Array2<f64>, Array1<f64>) {
    let random_array_a = array![
        [
            9.611_026_155_476_322,
            8.871_116_855_894_37,
            8.211_011_366_377_516,
            7.621_082_704_720_055,
            7.092_939_797_905_212,
            6.619_262_476_847_601,
            6.193_659_100_250_106,
            5.810_543_643_147_376
        ],
        [
            8.871_116_855_894_37,
            8.211_011_366_377_514,
            7.621_082_704_720_055,
            7.092_939_797_905_212,
            6.619_262_476_847_601,
            6.193_659_100_250_106,
            5.810_543_643_147_375,
            5.465_029_534_777_697
        ],
        [
            8.211_011_366_377_516,
            7.621_082_704_720_055,
            7.092_939_797_905_212,
            6.619_262_476_847_601,
            6.193_659_100_250_106,
            5.810_543_643_147_375,
            5.465_029_534_777_699,
            5.152_837_915_031_511
        ],
        [
            7.621_082_704_720_055,
            7.092_939_797_905_212,
            6.619_262_476_847_601,
            6.193_659_100_250_106,
            5.810_543_643_147_375,
            5.465_029_534_777_699,
            5.152_837_915_031_511,
            4.870_218_308_207_042_5
        ],
        [
            7.092_939_797_905_212,
            6.619_262_476_847_601,
            6.193_659_100_250_106,
            5.810_543_643_147_375,
            5.465_029_534_777_698,
            5.152_837_915_031_511,
            4.870_218_308_207_042_5,
            4.613_879_995_136_789
        ],
        [
            6.619_262_476_847_601,
            6.193_659_100_250_106,
            5.810_543_643_147_375,
            5.465_029_534_777_699,
            5.152_837_915_031_511,
            4.870_218_308_207_042_5,
            4.613_879_995_136_789,
            4.380_932_606_745_861
        ],
        [
            6.193_659_100_250_106,
            5.810_543_643_147_375,
            5.465_029_534_777_699,
            5.152_837_915_031_511,
            4.870_218_308_207_042_5,
            4.613_879_995_136_789,
            4.380_932_606_745_861,
            4.168_834_669_584_863_5
        ],
        [
            5.810_543_643_147_376,
            5.465_029_534_777_697,
            5.152_837_915_031_511,
            4.870_218_308_207_042_5,
            4.613_879_995_136_789,
            4.380_932_606_745_861,
            4.168_834_669_584_863_5,
            3.975_349_011_820_167
        ]
    ];

    let random_array_b = array![
        6.292_528_486_334_608,
        5.899_123_987_164_679,
        5.544_518_564_478_87,
        5.224_283_191_432_115,
        4.934_536_993_948_907,
        4.671_875_931_227_347,
        4.433_311_073_124_163,
        4.216_215_151_652_864_5
    ];

    (random_array_a, random_array_b)
}
