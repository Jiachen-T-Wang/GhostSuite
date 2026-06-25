"""
Data Value Embedding — reverse-order recursion over captured projected gradients.

Given per-step, per-sample projected gradients captured along a training trajectory
(``proj_iter_*.pt`` written by ``GradProjLoraEngine``), this module replays them in
**reverse** training order and produces one *data value embedding* per training point:

    e_s = g_s - M_{>s} g_s            (per sample)
    M_{>s-1} = M_{>s} + (lr_s/B_s) * sum_b e_{s,b} g_{s,b}^T

where ``M`` is maintained **per layer** (block-diagonal across layers), matching the
reference implementation (arXiv:2412.09538). The recursion is the unrolled first-order
SGD influence: e_s is g_s after being transformed by all later steps' gradient
outer-products.

Learning-rate handling is configurable via ``lr_mode``:
  - ``"none"``   : lr_s = 1, no 1/B_s normalization. Reproduces the released reference
                   code exactly (which drops the dynamic learning rate).
  - ``"scaled"`` : folds the recorded per-step learning rate into the embedding
                   (e_hat = lr_s * e_s) and divides the M accumulation by the batch
                   size (Gauss-Newton H ~= (1/B) sum_b g g^T).

The final value of a test point against a training point is ``<g_test, e_s>`` (computed
in ``dve_value.py``); the sign convention follows the reference (positive dot product).
"""

import json
from pathlib import Path
from typing import List, Optional, Sequence

import torch


def _reverse_step(M: Optional[torch.Tensor], g: torch.Tensor, lr: float,
                  lr_mode: str):
    """One reverse-recursion step for a single layer block.

    Args:
        M: running accumulation ``M_{>s}`` of size ``[D, D]`` (float32), or ``None``
           for the first processed (i.e. last training) step.
        g: projected per-sample gradients for this step/layer, shape ``[B, D]`` (float32).
        lr: learning rate recorded at this step.
        lr_mode: ``"none"`` or ``"scaled"`` (see module docstring).

    Returns:
        (emb, M_updated) where ``emb`` is ``[B, D]`` and ``M_updated`` is ``[D, D]``.
    """
    if M is None:
        e_raw = g
    else:
        # Per sample: (M g_b). flattened_grad @ M^T == [B,D] -> row_b = M g_b.
        e_raw = g - g @ M.t()

    if lr_mode == 'scaled':
        emb = lr * e_raw
        update = (emb.t() @ g) / g.shape[0]
    elif lr_mode == 'none':
        emb = e_raw
        update = emb.t() @ g
    else:
        raise ValueError(f"lr_mode must be 'none' or 'scaled', got {lr_mode!r}")

    M = update if M is None else M + update
    return emb, M


def dve_recursion(grads_by_step: Sequence[torch.Tensor],
                  lrs: Optional[Sequence[float]] = None,
                  lr_mode: str = 'none') -> List[torch.Tensor]:
    """Pure in-memory recursion over a single gradient block (one layer).

    Processes steps in reverse and returns embeddings in **training order**. Used by
    the unit test and small in-memory checks; the disk-streaming, per-layer version is
    :func:`compute_embeddings_reverse`.

    Args:
        grads_by_step: list of ``[B, D]`` tensors, in training order (step 0 first).
        lrs: optional per-step learning rates (same length); defaults to all 1.0.
        lr_mode: ``"none"`` or ``"scaled"``.
    """
    n = len(grads_by_step)
    M = None
    embs: List[Optional[torch.Tensor]] = [None] * n
    for idx in reversed(range(n)):
        lr = 1.0 if lrs is None else float(lrs[idx])
        emb, M = _reverse_step(M, grads_by_step[idx].float(), lr, lr_mode)
        embs[idx] = emb
    return embs  # type: ignore[return-value]


def _load_metadata(proj_dir: Path) -> dict:
    meta_path = proj_dir / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {proj_dir}; was capture run?")
    with open(meta_path) as f:
        return json.load(f)


def _list_proj_files(proj_dir: Path, ascending: bool) -> List[Path]:
    files = sorted(proj_dir.glob('proj_iter_*.pt'))
    if not files:
        raise FileNotFoundError(f"No proj_iter_*.pt files in {proj_dir}")
    return files if ascending else list(reversed(files))


def compute_embeddings_reverse(proj_dir, embed_dir, lr_mode: str = 'none',
                               device: str = 'cpu',
                               embed_dtype: torch.dtype = torch.float32,
                               verbose: bool = True) -> int:
    """Stream ``proj_iter_*.pt`` in reverse training order; write ``embed_iter_*.pt``.

    Maintains one block-diagonal ``M`` per layer (sliced from the concatenated
    projection via ``metadata.json``). Each output file mirrors the input's identity
    fields (``batch_idx``, ``iter``, ``lr``, ``order``) with ``proj`` replaced by the
    computed ``embedding``.

    Returns the number of files processed.
    """
    proj_dir = Path(proj_dir)
    embed_dir = Path(embed_dir)
    embed_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_metadata(proj_dir)
    layers = meta['layers']  # ordered list with slice_start / slice_end
    slices = [(L['slice_start'], L['slice_end']) for L in layers]

    files = _list_proj_files(proj_dir, ascending=False)  # reverse training order
    M_blocks: List[Optional[torch.Tensor]] = [None] * len(slices)

    # Carry the metadata forward so the embedding dir is self-describing.
    embed_meta = dict(meta)
    embed_meta['stage'] = 'embedding'
    embed_meta['lr_mode'] = lr_mode
    with open(embed_dir / 'metadata.json', 'w') as f:
        json.dump(embed_meta, f, indent=2)

    for n_done, f in enumerate(files):
        d = torch.load(f, map_location=device)
        proj = d['proj'].to(device=device, dtype=torch.float32)  # [B, total]
        lr = float(d.get('lr', 1.0))

        emb_full = torch.empty_like(proj)
        for i, (s, e) in enumerate(slices):
            g = proj[:, s:e]
            emb, M_blocks[i] = _reverse_step(M_blocks[i], g, lr, lr_mode)
            emb_full[:, s:e] = emb

        save_dict = {
            'embedding': emb_full.to(embed_dtype).cpu(),
            'iter': int(d.get('iter', 0)),
            'batch_size': int(proj.shape[0]),
            'lr': lr,
        }
        if 'batch_idx' in d:
            save_dict['batch_idx'] = d['batch_idx']
        if 'order' in d:
            save_dict['order'] = int(d['order'])

        out_name = f"embed_iter_{int(d.get('iter', n_done)):06d}.pt"
        torch.save(save_dict, embed_dir / out_name)

        if verbose and (n_done % 50 == 0 or n_done == len(files) - 1):
            print(f"  [embedding] {n_done + 1}/{len(files)} "
                  f"(file {f.name} -> {out_name})")

    print(f"[embedding] wrote {len(files)} embedding files to {embed_dir} "
          f"(lr_mode={lr_mode})")
    return len(files)
