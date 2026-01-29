//! End-to-end benchmarks: full generation and throughput.

use bitnet_oxidized::{create_demo_model, InferenceEngine, TextGenerator};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_full_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_generation");

    let model = create_demo_model();
    let generator = TextGenerator::new(model);
    let prompt = vec![0usize, 1, 2];

    for max_new in [10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("greedy", max_new),
            &max_new,
            |b, &max_new| {
                b.iter(|| {
                    generator
                        .generate_greedy(&prompt, prompt.len() + max_new)
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_forward_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_latency");

    let model = create_demo_model();
    let engine = InferenceEngine::new(model);

    for seq_len in [1, 4, 8, 16] {
        let input: Vec<usize> = (0..seq_len).map(|i| i % 256).collect();
        group.bench_with_input(BenchmarkId::new("forward", seq_len), &seq_len, |b, _| {
            b.iter(|| engine.forward(&input).unwrap());
        });
    }

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    let model = create_demo_model();
    let engine = InferenceEngine::new(model);

    for batch_size in [1, 2, 4, 8] {
        let batch: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0, 1, 2]).collect();
        group.bench_with_input(
            BenchmarkId::new("forward_batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| engine.forward_batch(&batch).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_full_generation,
    bench_forward_latency,
    bench_throughput
);
criterion_main!(benches);
