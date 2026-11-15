"""Perceptual hashes (aHash, pHash) and near-duplicate grouping (proposal 3.4).

Hashes are 64-bit, stored in the manifest as 16-char hex strings. Near copies are pairs
within a Hamming cut-off; chains of near copies are joined into groups with union-find.
"""
from __future__ import annotations

import numpy as np

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def compute_hashes(rgb: np.ndarray) -> tuple[str, str]:
    import imagehash
    from PIL import Image
    img = Image.fromarray(rgb)
    return str(imagehash.average_hash(img)), str(imagehash.phash(img))


def hex_to_u64(hexes) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances: a (n,) x b (m,) uint64 -> (n, m)."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def _close(da, dp, a_max, p_max, mode):
    return ((da <= a_max) & (dp <= p_max)) if mode == "both" else ((da <= a_max) | (dp <= p_max))


def near_pairs(ah, ph, a_max=4, p_max=6, mode="both", chunk=256) -> np.ndarray:
    """All index pairs (i<j) within one set that are near copies. Returns (M, 2) int array."""
    n, out = len(ah), []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        ok = _close(hamming(ah[s:e], ah), hamming(ph[s:e], ph), a_max, p_max, mode)
        ii, jj = np.nonzero(ok)
        ii = ii + s
        keep = ii < jj
        out.append(np.stack([ii[keep], jj[keep]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def group_ids(n: int, pairs: np.ndarray) -> np.ndarray:
    """Union-find over near-copy pairs -> contiguous group id per image."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def pair_distances(ah, ph, pairs) -> np.ndarray:
    """(M,2) array of [aHash dist, pHash dist] for given pairs: use to pick the cut-off by eye."""
    da = np.array([bin(int(ah[i] ^ ah[j])).count("1") for i, j in pairs])
    dp = np.array([bin(int(ph[i] ^ ph[j])).count("1") for i, j in pairs])
    return np.stack([da, dp], 1)
