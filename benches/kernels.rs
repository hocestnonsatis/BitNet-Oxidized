//! Criterion benchmarks for mat-vec kernels.

use bitnet_oxidized::kernels::{
    mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, TernaryTensor,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

fn make_weights(out: usize, inp: usize) -> (TernaryTensor, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut t = TernaryTensor::zeros(out * inp);
    for i in 0..(out * inp) {
        t.set(i, rng.gen_range(-1..=1) as f32);
    }
    let input: Vec<f32> = (0..inp).map(|_| rng.gen()).collect();
    (t, input)
}

fn bench_matvec(c: &mut Criterion) {
    let (weight, input) = make_weights(256, 512);
    let mut out = vec![0.0f32; 256];

    c.bench_function("matvec_basic_256x512", |b| {
        b.iter(|| mat_vec_mul_basic(black_box(&weight), black_box(&input), black_box(&mut out)))
    });
    c.bench_function("matvec_blocked_256x512", |b| {
        b.iter(|| mat_vec_mul_blocked(black_box(&weight), black_box(&input), black_box(&mut out)))
    });
    c.bench_function("matvec_lut_256x512", |b| {
        b.iter(|| mat_vec_mul_lut(black_box(&weight), black_box(&input), black_box(&mut out)))
    });
}

criterion_group!(benches, bench_matvec);
criterion_main!(benches);
