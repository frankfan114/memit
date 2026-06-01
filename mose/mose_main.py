from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from memit.compute_ks import compute_ks
from memit.compute_z import compute_z, get_module_input_output_at_words
from memit.memit_main import get_cov, get_context_templates

from .mose_hparams import MOSEHyperParams


def apply_mose_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MOSEHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Apply MOSE edits to the model.  For each rewrite layer solves an orthogonal
    Procrustes problem to find R ∈ O(d) and sets W_new = R @ W0.
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    new_weights = execute_mose(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, W_new in new_weights.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] = W_new.to(w.device).to(w.dtype)

    print(f"New weights successfully inserted into {list(new_weights.keys())}")
    return model, weights_copy


def execute_mose(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MOSEHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Core MOSE algorithm.

    For each rewrite layer i with original weight W0 ∈ R^{d_out × d_in}:

      1.  K_E = compute_ks(...)                  [d_in, n_edit]
      2.  V_E = W0 @ K_E + resid_i               [d_out, n_edit]
             where resid_i distributes the z-space error across layers
      3.  C0  = get_cov(...)                      [d_in, d_in]   (E[kk^T])
      4.  M   = V_E (W0 K_E)^T + λ W0 C0 W0^T   [d_out, d_out]
      5.  M   = U Σ V^T   (SVD)
      6.  R   = U V^T                             R ∈ O(d_out)
      7.  W_new = R W0

    Invariant: model weights are restored to their original values before return.
    Returns: dict  weight_name → W_new  (CPU, float32).
    """
    new_weights: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Preprocess requests (same as MEMIT)
    # ------------------------------------------------------------------ #
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MOSE request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # ------------------------------------------------------------------ #
    # Snapshot weights that will be modified
    # ------------------------------------------------------------------ #
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # ------------------------------------------------------------------ #
    # Compute target vectors z* via compute_z (identical to MEMIT)
    # ------------------------------------------------------------------ #
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
            z_list.append(cur_z)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                print(f"Cached z vector at {cache_fname}")

    zs = torch.stack(z_list, dim=1)  # [d_out, n_requests]

    # ------------------------------------------------------------------ #
    # Per-layer Procrustes solve
    # ------------------------------------------------------------------ #
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # -- Edit keys K_E ------------------------------------------------
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        # layer_ks: [d_in, n_edit]
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # -- Residual error at z_layer (progressive, same as MEMIT) ------
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[req["prompt"] for req in requests],
            words=[req["subject"] for req in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T  # [d_out, n_requests]

        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)  # [d_out, n_edit]
        resid = targets / (len(hparams.layers) - i)                # distribute across layers

        # -- Load preserved-key second moment C0 --------------------------
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        )  # C0: [d_in, d_in]

        # -- Retrieve W0 in mathematical [d_out, d_in] orientation --------
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        W0_stored = weights_copy[weight_name]  # original weight (on cuda)
        d_in = layer_ks.shape[0]

        if W0_stored.shape[1] == d_in:
            # Standard PyTorch layout: [d_out, d_in]
            W0 = W0_stored
            stored_transposed = False
        elif W0_stored.shape[0] == d_in:
            # Transposed layout: [d_in, d_out]
            W0 = W0_stored.T
            stored_transposed = True
        else:
            raise ValueError(
                f"Weight {weight_name} shape {W0_stored.shape} is incompatible "
                f"with d_in={d_in} inferred from key vectors."
            )

        # -- Procrustes solve in double precision -------------------------
        W0     = W0.double()
        KE     = layer_ks.double()         # [d_in, n_edit]
        resid  = resid.double()            # [d_out, n_edit]
        C0     = cov.double()              # [d_in, d_in]
        lam    = hparams.mom2_update_weight

        W0_KE      = W0 @ KE                              # [d_out, n_edit]
        VE         = W0_KE + resid                        # [d_out, n_edit]

        edit_term  = VE @ W0_KE.T                         # [d_out, d_out]
        pres_term  = lam * (W0 @ C0 @ W0.T)               # [d_out, d_out]
        M          = edit_term + pres_term                 # [d_out, d_out]

        U, _, Vh   = torch.linalg.svd(M, full_matrices=False)
        R          = U @ Vh                               # R ∈ O(d_out)

        W0_new     = (R @ W0).float()                     # [d_out, d_in], float32

        print("orig norm", torch.linalg.norm(W0_stored).item())
        print("R  det  ", torch.linalg.det(R.float()).item())
        print("new  norm", torch.linalg.norm(W0_new).item())

        # Restore storage orientation
        W0_new_stored = W0_new.T.contiguous() if stored_transposed else W0_new

        # Apply update in-place so subsequent layers see the updated model
        with torch.no_grad():
            weights[weight_name][...] = W0_new_stored

        new_weights[weight_name] = W0_new_stored.cpu()

        # Free GPU memory
        cov.cpu()
        del W0, KE, resid, C0, W0_KE, VE, edit_term, pres_term, M, U, Vh, R, W0_new
        del layer_ks, cur_zs, targets
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Restore original model state (invariant)
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"MOSE weight updates computed for {list(new_weights.keys())}")
    return new_weights
