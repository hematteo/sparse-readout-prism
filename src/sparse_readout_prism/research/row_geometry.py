"""Row-geometry helpers shared by the direct-geometry baselines and the stability scripts.

* :func:`spherical_kmeans_unit` -- Lloyd's k-means on unit-norm rows with cosine
  assignment, the clustering behind the ``row_cluster_*`` baselines
  (``scripts/run/run_readout_baseline_comparisons.py``) and the leave-one-out
  core-recovery control (``scripts/eval/loo_core_recovery.py``). Seeded
  initialisation and reseeding; bit-exact across runs on CPU only (CUDA
  ``index_add_`` accumulates in an order-dependent way, so GPU centroids can
  differ at the last ulp between runs).
* :func:`resolve_single_token_bare_first` -- the bare-then-space-prefixed
  single-token resolver used by the causal-validation and stability scripts.
  The fidelity runners use the space-first, special-rejecting
  :func:`sparse_readout_prism.research.registry.resolve_single_token_strict`;
  the two orders can pick different rows for terms where both variants are
  single tokens, so scripts state which one they use.
"""

from __future__ import annotations

import torch


def spherical_kmeans_unit(
    X: torch.Tensor,
    n_clusters: int,
    seed: int,
    *,
    iters: int = 12,
    chunk: int = 4096,
    return_assignments: bool = False,
    final_assignment: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Unit-normalised centroids ``(n_clusters, d)`` for unit-norm rows ``X`` ``(V, d)``.

    Dead centroids are reseeded from random rows each iteration. With
    ``return_assignments=True`` an assignment ``(V,)`` is returned as well:
    the cosine argmax against the returned centroids (``final_assignment=True``,
    the ``row_cluster_*`` baselines' reading), or, with
    ``final_assignment=False``, the assignment that produced the last centroid
    update -- one Lloyd step behind the returned centroids. The paper's
    leave-one-out cluster control was computed with the latter, so
    ``loo_core_recovery.py`` keeps it; the two differ only while the iteration
    has not converged.
    """
    V = X.shape[0]
    if n_clusters > V:
        raise ValueError(f"n_clusters={n_clusters} exceeds the number of rows V={V}")
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(V, generator=g)[:n_clusters].to(X.device)].clone()
    ones = torch.ones(V, device=X.device)
    assign = torch.empty(V, dtype=torch.long, device=X.device)
    for _ in range(iters):
        for s in range(0, V, chunk):
            assign[s : s + chunk] = (X[s : s + chunk] @ C.T).argmax(dim=1)
        C_new = torch.zeros_like(C)
        count = torch.zeros(n_clusters, device=X.device)
        C_new.index_add_(0, assign, X)
        count.index_add_(0, assign, ones)
        dead = count == 0
        C = C_new / count.clamp_min(1.0)[:, None]
        n_dead = int(dead.sum())
        if n_dead:
            ridx = torch.randperm(V, generator=g)[:n_dead].to(X.device)
            C[dead] = X[ridx]
        C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if return_assignments:
        if final_assignment:
            for s in range(0, V, chunk):
                assign[s : s + chunk] = (X[s : s + chunk] @ C.T).argmax(dim=1)
        return C, assign
    return C


def resolve_single_token_bare_first(tok, term: str) -> int | None:
    """Single token id for ``term``, trying the bare form before the space-prefixed form."""
    for variant in (term, " " + term):
        ids = tok.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None
