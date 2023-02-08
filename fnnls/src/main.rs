use fnnls::fnnls::*;

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