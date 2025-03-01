import torch
import torch.nn.functional as F



def vector_quantization_distillation_loss(student_logits, teacher_logits, student_token_embeddings, beta_vq=0.25, beta_distill=1.0):
    """
    Calculates a distillation loss using Vector Quantization (VQ) 
    to compare teacher logits with student logits.

    Args:
        student_logits: Logits from the student model. Shape: (batch_size, sequence_length, hidden_size)
        teacher_logits: Logits from the teacher model. Shape: (batch_size, sequence_length, hidden_size)
        student_token_embeddings: Token embedding matrix of the student model (codebook). Shape: (student_vocab_size, embedding_dim)
        beta_vq: Commitment cost for VQ loss (teacher's logits quantization).
        beta_distill: Weight for the distillation loss component.

    Returns:
        total_loss: Combined loss value (VQ loss + distillation loss).
        quantized_teacher_logits: Quantized teacher logits.
        teacher_encoding_indices: Encoding indices from teacher logits quantization.
        distillation_loss_val: Value of the distillation loss component.
        vq_loss_val: Value of the VQ loss component.
    """

    # 1. Vector Quantization of Teacher Logits (using student's token embeddings as codebook)
    # Re-use the vector_quantization_loss function from the previous example
    vq_loss_val, quantized_teacher_logits, teacher_encoding_indices = vector_quantization_loss_for_distillation(
        teacher_logits, student_token_embeddings, beta=beta_vq
    )

    # 2. Distillation Loss:  MSE between Student Logits and Quantized Teacher Logits
    distillation_loss_val = F.mse_loss(student_logits, quantized_teacher_logits.detach()) # Detach quantized teacher logits

    # 3. Total Loss: Combine VQ loss (on teacher logits) and distillation loss
    total_loss = beta_distill * distillation_loss_val + vq_loss_val

    return total_loss, quantized_teacher_logits, teacher_encoding_indices, distillation_loss_val, vq_loss_val


def vector_quantization_loss_for_distillation(logits, token_embeddings, beta=0.25):
    """
    (Reused and slightly renamed from previous example for clarity in distillation context)
    Calculates the Vector Quantization (VQ) loss for a given logits tensor 
    using provided token embeddings as the codebook.

    Args:
        logits: Tensor of logits to be quantized. Shape: (batch_size, sequence_length, hidden_size)
        token_embeddings: Token embedding matrix (codebook). Shape: (vocab_size, embedding_dim)
        beta: Commitment cost.

    Returns:
        vq_loss: Scalar VQ loss value.
        quantized_logits: Tensor of quantized logits.
        encoding_indices: Indices of the closest token embeddings.
    """
    # --- (Implementation of vector_quantization_loss function from previous answer is copied here) ---
    # 1. Flatten logits for distance calculation
    flattened_logits = logits.view(-1, logits.shape[-1])

    # 2. Calculate distances between flattened logits and all token embeddings (codebook)
    distances = torch.cdist(flattened_logits.float(), token_embeddings.float(), p=2)

    # 3. Encoding: Find the index of the closest token embedding for each logit vector
    encoding_indices = torch.argmin(distances, dim=-1)

    # 4. Quantize: Get the quantized logits by gathering the closest token embeddings
    quantized_logits_flat = torch.index_select(token_embeddings, dim=0, index=encoding_indices)

    # 5. Reshape quantized logits back to original logit shape
    quantized_logits = quantized_logits_flat.view_as(logits)

    # 6. Commitment Loss: L2 loss between original logits and quantized logits
    commitment_loss = F.mse_loss(quantized_logits.detach(), logits)

    # 7. VQ Loss: Weighted commitment loss
    vq_loss = beta * commitment_loss

    return vq_loss, quantized_logits, encoding_indices
