import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def get_probe_logits(inputs, subject, probe):
    with torch.no_grad():
        activations = subject(inputs)
    logits = probe(activations)
    return logits


def train_language_probe(
    dataloader,
    probe,
    teacher: nn.Module,
    tokenizer,
    epochs,
    log_dir,
):
    device = next(teacher.parameters()).device
    probe = probe.to(device)
    writer = SummaryWriter(log_dir)

    freeze_model(teacher)

    optimizer = torch.optim.Adam(probe.parameters())

    # Set up logging
    logging.basicConfig(
        filename=f"{log_dir}/probe_log.txt", level=logging.INFO, format="%(message)s"
    )
    logger = logging.getLogger()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (inputs, activations) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        ):
            logits = probe(activations.to(device))
            indices = logits.argmax(dim=-1)
            # import pdb; pdb.set_trace()
            # print(token_embeds.shape)
            teacher_outputs = teacher(input_ids=indices)
            # print(teacher_outputs)
            teacher_logits = teacher_outputs.logits

            loss = nn.functional.cross_entropy(logits, teacher_logits.softmax(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Print the probe's tokens from a random input
        input, activation = next(iter(dataloader))
        logits = probe(activation)
        tokens = logits.argmax(dim=-1)
        text = tokenizer.decode(tokens[0])

        input_str = f"Input: {input}\n"
        probe_str = f"Probe: {text}\n"
        loss_str = f"Loss: {avg_loss}\n"

        print(input_str)
        print(probe_str)
        print(loss_str)
        logger.info(input_str)
        logger.info(probe_str)
        logger.info(loss_str)

        # Log input and probe to tensorboard
        writer.add_text("Input", input_str, epoch)
        writer.add_text("Probe", probe_str, epoch)

    writer.close()
    torch.save(probe.state_dict(), f"{log_dir}/probe_state_dict.pth")
    return probe

def train_linear_language_probe(
    dataloader,
    teacher: nn.Module,
    tokenizer,
    embed_dim,
    vocab_size,
    epochs,
    log_dir,
):
    probe = nn.Linear(embed_dim, vocab_size)
    return train_language_probe(dataloader, probe, teacher, tokenizer, epochs, log_dir)