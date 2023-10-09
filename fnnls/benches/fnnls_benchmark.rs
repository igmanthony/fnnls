#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fnnls::fnnls::*;

use std::cell::RefCell;

fn criterion_benchmark(c: &mut Criterion) {
    let (RtR, Rtb) = fnnls::test_setup();
    let RtR = RefCell::new(RtR);
    let Rtb = RefCell::new(Rtb);
    c.bench_function("fnnls", move |b| {
        b.iter(|| {
            let rtr = RtR.borrow();
            let rtb = Rtb.borrow();
            fnnls(black_box(&rtr), black_box(&rtb));
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
