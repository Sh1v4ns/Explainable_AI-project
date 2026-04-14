# viashap_loss.py
import torch
import torch.nn.functional as F

def sample_coalition_mask(batch_size: int, n_features: int, device) -> torch.Tensor:
    """
    Sample a random binary coalition mask S for each instance.
    Uses the Shapley kernel weighting implicitly via uniform sampling.
    
    Returns: (batch, n_features) binary mask
    """
    mask = torch.randint(0, 2, (batch_size, n_features), dtype=torch.float32, device=device)
    return mask


def viashap_loss(
    model,
    x:       torch.Tensor,   # (batch, n_features)
    y:       torch.Tensor,   # (batch,) — class labels
    beta:    float = 10.0,
    n_samples: int = 32,
    baseline: str = 'zero',  # paper default: baseline removal
):
    """
    Computes the dual-objective ViaSHAP loss (Equation 7 in paper).

    L = L_pred + beta * L_phi

    L_phi = E_S [ (ViaSHAP(x_S) - ViaSHAP(x_0) - 1_S^T phi(x))^2 ]
    """
    batch_size, n_features = x.shape
    device = x.device

    # --- Baseline: zero vector (baseline removal approach) ---
    x_baseline = torch.zeros_like(x)

    total_phi_loss = torch.tensor(0.0, device=device)

    for _ in range(n_samples):
        # 1. Sample coalition mask
        S = sample_coalition_mask(batch_size, n_features, device)  # (batch, n_features)

        # 2. Masked input: keep features in S, set others to baseline
        x_S = x * S + x_baseline * (1 - S)                        # (batch, n_features)

        # 3. Forward passes
        phi,  y_hat   = model(x)      # full input  → Shapley + prediction
        phi_S, y_S    = model(x_S)    # masked input → prediction only needed
        _,     y_base = model(x_baseline.expand_as(x))  # baseline prediction

        # 4. ViaSHAP(x_S) - ViaSHAP(x_0): scalar per (batch, class)
        diff = y_S - y_base           # (batch, n_classes)

        # 5. 1_S^T phi: sum Shapley values of selected features
        #    phi shape: (batch, n_features, n_classes)
        #    S shape:   (batch, n_features) → unsqueeze for broadcast
        phi_selected = (phi * S.unsqueeze(-1)).sum(dim=1)  # (batch, n_classes)

        # 6. Shapley loss per sample
        phi_loss = ((diff - phi_selected) ** 2).mean()
        total_phi_loss = total_phi_loss + phi_loss

    total_phi_loss = total_phi_loss / n_samples

    # --- Prediction loss (cross-entropy) ---
    _, y_hat_full = model(x)
    pred_loss = F.cross_entropy(y_hat_full, y.long())

    total_loss = pred_loss + beta * total_phi_loss
    return total_loss, pred_loss.item(), total_phi_loss.item()