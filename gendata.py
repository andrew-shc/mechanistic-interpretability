import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import multiprocessing
import json
import os
from tqdm import tqdm


@torch.inference_mode()
def generate_reasoning_trace_with_activations(prompt, model, tokenizer):
    """

    stacked_hidden_states: A tensor of shape (num_layers, batch_size, sequence_length, hidden_size)
    """
    prompt = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

    outputs = model.generate(
        **inputs,
        output_scores=True,
        output_hidden_states=True,  # Get hidden states
        return_dict_in_generate=True,
        temperature=0.6,
        top_p=0.95,
        max_length=512,
    )
    generated_ids = outputs.sequences
    decoded_output = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
    hidden_states = outputs.hidden_states  # Tuple of hidden states (layers)

    # Concatenate hidden states along the sequence length dimension
    concatenated_layers = []
    for layer_states in zip(*hidden_states):  # This will iterate through each layer
        concatenated_steps = []
        for (
            step_state
        ) in layer_states:  # this will iterate through each generation step.
            concatenated_steps.append(step_state)
        concatenated_layers.append(
            torch.cat(concatenated_steps, dim=1)
        )  # concatenate the steps.

    # Stack the layers into a single tensor
    stacked_hidden_states = torch.stack(concatenated_layers, dim=0)

    return decoded_output, stacked_hidden_states.to("cpu")


if __name__ == "__main__":
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    prompts = [p["prompt"] for p in prompts]
    # prompts = [[{"role": "user", "content": p}] for p in prompts]
    output_dir = "data"  # Directory to save files.
    os.makedirs(
        output_dir, exist_ok=True
    )  # creates the directory if it does not exist.
    results = []
    for i, p in enumerate(tqdm(prompts, desc="Generating traces")):
        decoded_output, hidden_states = generate_reasoning_trace_with_activations(
            p, model, tokenizer
        )
        print(decoded_output)

        output_file = os.path.join(output_dir, f"trace_{i}.pt")
        torch.save(
            {
                "decoded_output": decoded_output,
                "hidden_states": hidden_states,
            },
            output_file,
        )
        print(f"Saved trace to {output_file}")
