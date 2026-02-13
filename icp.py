"""ICP - Iterative Closest Point"""

import numpy as np


def nearest_neighbors(src, tgt):
    """For each src point, find the nearest tgt point (vectorised)."""
    diff = src[:, None, :] - tgt[None, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    idx = np.argmin(dists_sq, axis=1)
    return idx, np.sqrt(dists_sq[np.arange(len(src)), idx])


def best_fit_transform(src, tgt):
    """SVD-based optimal rotation R and translation t."""
    cs, ct = src.mean(0), tgt.mean(0)
    H = (src - cs).T @ (tgt - ct)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, ct - R @ cs


def icp(source, target, max_iter=50, tol=1e-6):
    """Align source to target, yielding state each iteration."""
    src = source.copy()
    prev = float("inf")
    for i in range(1, max_iter + 1):
        idx, dists = nearest_neighbors(src, target)
        R, t = best_fit_transform(src, target[idx])
        src = (R @ src.T).T + t
        err = dists.mean()
        converged = abs(prev - err) < tol
        yield {"iteration": i, "transformed": src.copy(), "error": err, "converged": converged}
        if converged:
            return
        prev = err
