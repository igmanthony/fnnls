mod fnnls;
use fnnls::fnnls;

use ndarray::{array, prelude::*};

fn main() {
    let (a, b) = test_setup();
    dbg!(fnnls(&a, &b));
    // produces:
    // [0.0032405261283480506, 0.0, 0.0, 0.0, 0.0, 
    //  0.2290345680837889, 0.7661616026538993, 0.0]
    // and:
    // [0.0, -0.0000005656027068923208,  -0.0000005729504790608075,
    //       -0.00000036045811047102916, -0.00000013335828796812166,
    //  0.0, 0.0, -0.0000001258811650473035]

    // updated Python script produces:
    // [0.00324053, 0.0, 0.0, 0.0, 0.0, 0.22903457, 0.7661616, 0.0]
    // [ 0.00000000e+00, -5.65602707e-07, -5.72950479e-07, -3.60458110e-07,
    //  -1.33358288e-07,  0.00000000e+00,  0.00000000e+00, -1.25881165e-07]

    // scipy.optimize.nnls produces:
    // [0.00331854, 0.0, 0.0, 0.0, 0.0, 0.22834084, 0.76678191, 0.0]
    // and:
    // 5.010905800409835e-07
}

fn test_setup() -> (Array2<f64>, Array1<f64>) {
    let random_array_a = array![
        [ 9.611026155476322,   8.871116855894370,   8.211011366377516,
          7.621082704720055,   7.092939797905212,   6.619262476847601,
          6.193659100250106,   5.810543643147376], [8.871116855894370,
          8.211011366377514,   7.621082704720055,   7.092939797905212,
          6.619262476847601,   6.193659100250106,   5.810543643147375,
          5.465029534777697], [8.211011366377516,   7.621082704720055,
          7.092939797905212,   6.619262476847601,   6.193659100250106,
          5.810543643147375,   5.465029534777699,   5.152837915031511],
         [7.621082704720055,   7.092939797905212,   6.619262476847601,
          6.193659100250106,   5.810543643147375,   5.465029534777699,
          5.152837915031511,   4.8702183082070425],[7.092939797905212,
          6.619262476847601,   6.193659100250106,   5.810543643147375,
          5.465029534777698,   5.152837915031511,   4.8702183082070425,
          4.613879995136789], [6.619262476847601,   6.193659100250106,
          5.810543643147375,   5.465029534777699,   5.152837915031511,
          4.8702183082070425,  4.613879995136789,   4.380932606745861],
         [6.193659100250106,   5.810543643147375,   5.465029534777699,
          5.152837915031511,   4.8702183082070425,  4.613879995136789,
          4.380932606745861,   4.1688346695848635],[5.810543643147376, 
          5.465029534777697,   5.152837915031511,   4.8702183082070425,
          4.613879995136789,   4.380932606745861,   4.1688346695848635,
          3.975349011820167]
        ];

    let random_array_b = array![
            6.292528486334608, 5.899123987164679, 5.54451856447887,
            5.224283191432115, 4.934536993948907, 4.671875931227347,
            4.433311073124163, 4.2162151516528645
        ];

    (random_array_a, random_array_b)                                       
}