---
library_name: transformers
base_model:
- meta-llama/Meta-Llama-3-8B-Instruct
---

# Model Card for Model ID

### Llama3-8B-1.58 Models

The **Llama3-8B-1.58** models are large language models fine-tuned on the **BitNet 1.58b architecture**, starting from the base model **Llama-3-8B-Instruct**.

For a deeper dive into the methods and results, check out our [blog post](https://huggingface.co/blog/1_58_llm_extreme_quantization).


## Model Details

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [Model](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)
- **Paper:** [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)


## How to Get Started with the Model

You can easily load and test our model in Transformers. Just follow the code below:

Start by installing the transformers version with the correct configuration to load bitnet models
```bash
pip install git+https://github.com/huggingface/transformers.git@refs/pull/33410/head
```
And then load the model : 
```python

model = AutoModelForCausalLM.from_pretrained("HF1BitLLM/Llama3-8B-1.58-100B-tokens", device_map="cuda", torch_dtype=torch.bfloat16)    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

input_text = "Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:"

input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
output = model.generate(input_ids, max_length=10, do_sample=False)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## Training Details

### Training Data

The model was trained on a subset of [FineWeb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

### Training Process

1. **Starting Point**
   - Best-performing checkpoint from the 10 billion token runs with a linear lambda scheduler

2. **Training Duration**
   - Fine-tuned for an additional 45,000 steps
   - Reached a total of 100 billion tokens

3. **Dataset**
   - FineWeb-edu dataset

4. **Batch Size**
   - 2 million tokens per step
   - Total per run: 45,000 steps * 2 million tokens = 90 billion tokens
   - Combined with initial 10 billion tokens to reach 100 billion

5. **Learning Rate Experiments**
   - Tested various learning rates to find optimal setting, according the to experiments, the best performing peak lr is 1e-5

6. **Performance**
   - Close to Llama3 8B on some metrics
   - Behind Llama3 8B in overall average performance

7. **Evaluation**
   - Metrics included perplexity, MMLU scores, and other standard benchmarks

These extended training runs on 100 billion tokens pushed the boundaries of highly quantized models, bringing performance closer to half-precision models like Llama3.


## Evaluation

The evaluation of the models is done on the nanotron checkpoints using LightEval : 

![results](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metrics_100B_table.png)



## Citation

```bash
@misc{,
      title={1.58-Bit LLM: A New Era of Extreme Quantization}, 
      author={Mohamed Mekkouri and Marc Sun and Leandro von Werra and Thomas Wolf},
      year={2024},
}
```