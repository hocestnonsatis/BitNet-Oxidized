//! Advanced generation: pipeline with logit processors and sampling strategies.
//!
//! Demonstrates:
//! - GenerationPipeline with RepetitionPenalty, TopK, TopP
//! - Sampling strategies: Greedy, TopK, TopP, Mirostat, LocallyTypical
//! - ProcessorChain composition

use anyhow::Result;
use bitnet_oxidized::{
    create_demo_model,
    inference::{
        logit_processors::{
            ProcessorChain, RepetitionPenaltyProcessor, TemperatureProcessor, TopKProcessor,
        },
        pipeline::{GenerationPipeline, SamplingStrategy},
    },
};

fn main() -> Result<()> {
    let model = create_demo_model();
    let prompt = vec![0usize, 1, 2];

    // Pipeline with repetition penalty + top-k + temperature (chain)
    let chain = ProcessorChain::new()
        .add_processor(RepetitionPenaltyProcessor::new(1.2))
        .add_processor(TopKProcessor::new(50))
        .add_processor(TemperatureProcessor::new(0.7));

    let pipeline = GenerationPipeline::new(model.clone())
        .chain(chain)
        .with_strategy(SamplingStrategy::TopP {
            p: 0.9,
            temperature: 0.7,
        })
        .max_tokens(20)
        .eos_token_id(Some(2));

    let ids = pipeline.generate(&prompt)?;
    println!(
        "Pipeline (TopP) generated {} tokens: {:?}",
        ids.len(),
        &ids[..ids.len().min(15)]
    );

    // Mirostat sampling
    let pipeline_miro = GenerationPipeline::new(model.clone())
        .add_processor(RepetitionPenaltyProcessor::new(1.15))
        .with_strategy(SamplingStrategy::Mirostat { tau: 3.0, eta: 0.1 })
        .max_tokens(15);

    let ids_miro = pipeline_miro.generate(&prompt)?;
    println!(
        "Mirostat generated {} tokens: {:?}",
        ids_miro.len(),
        &ids_miro[..ids_miro.len().min(12)]
    );

    // Locally typical sampling
    let pipeline_typical = GenerationPipeline::new(model.clone())
        .add_processor(RepetitionPenaltyProcessor::new(1.1))
        .with_strategy(SamplingStrategy::LocallyTypical {
            p: 0.9,
            temperature: 0.8,
        })
        .max_tokens(15);

    let ids_typical = pipeline_typical.generate(&prompt)?;
    println!(
        "LocallyTypical generated {} tokens: {:?}",
        ids_typical.len(),
        &ids_typical[..ids_typical.len().min(12)]
    );

    // Greedy (deterministic)
    let pipeline_greedy = GenerationPipeline::new(model)
        .with_strategy(SamplingStrategy::Greedy)
        .max_tokens(10);

    let ids_greedy = pipeline_greedy.generate(&prompt)?;
    println!(
        "Greedy generated {} tokens: {:?}",
        ids_greedy.len(),
        ids_greedy
    );

    Ok(())
}
