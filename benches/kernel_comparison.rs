//! Kernel comparison: basic, blocked, LUT, SIMD at various matrix sizes.
//! Reports throughput (GFLOPS) for mat-vec: 2 * M * N ops (multiply-add per element).

use bitnet_oxidized::kernels::{
    mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, mat_vec_mul_simd, TernaryTensor,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;

fn make_weights(out: usize, inp: usize) -> (TernaryTensor, Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut t = TernaryTensor::zeros(out * inp);
    for i in 0..(out * inp) {
        t.set(i, rng.gen_range(-1..=1) as f32);
    }
    let input: Vec<f32> = (0..inp).map(|_| rng.gen()).collect();
    let output = vec![0.0f32; out];
    (t, input, output)
}

fn bench_kernels(c: &mut Criterion) {
    let sizes: Vec<(usize, usize)> = vec![
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ];

    let mut group = c.benchmark_group("matvec_kernels");
    group.sample_size(50);

    for (out, inp) in sizes {
        let (weight, input, mut out_buf) = make_weights(out, inp);

        group.bench_with_input(
            BenchmarkId::new("basic", format!("{}x{}", out, inp)),
            &(out, inp),
            |b, _| {
                b.iter(|| {
                    mat_vec_mul_basic(
                        black_box(&weight),
                        black_box(&input),
                        black_box(&mut out_buf),
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("blocked", format!("{}x{}", out, inp)),
            &(out, inp),
            |b, _| {
                b.iter(|| {
                    mat_vec_mul_blocked(
                        black_box(&weight),
                        black_box(&input),
                        black_box(&mut out_buf),
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("lut", format!("{}x{}", out, inp)),
            &(out, inp),
            |b, _| {
                b.iter(|| {
                    mat_vec_mul_lut(
                        black_box(&weight),
                        black_box(&input),
                        black_box(&mut out_buf),
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", out, inp)),
            &(out, inp),
            |b, _| {
                b.iter(|| {
                    mat_vec_mul_simd(
                        black_box(&weight),
                        black_box(&input),
                        black_box(&mut out_buf),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
