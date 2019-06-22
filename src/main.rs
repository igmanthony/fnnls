mod fnnls;
use fnnls::{fnnls, test_setup};


fn main() {
    let (RtR, Rtb) = test_setup();
    dbg!(fnnls(&RtR, &Rtb));
}

