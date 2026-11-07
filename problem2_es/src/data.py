"""Loading and validating the OR-Library portfolio datasets (port1-port5).

File format (https://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html):
    line 1          : number of assets n
    next n lines    : mean return, standard deviation of asset i
    remaining lines : i j correlation   (1-based indices, i <= j, diagonal included)

The covariance matrix is Sigma_ij = s_i * s_j * C_ij.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Expected (assets, correlation records) from the assignment table.
EXPECTED = {
    "port1": (31, 496),
    "port2": (85, 3655),
    "port3": (89, 4005),
    "port4": (98, 4851),
    "port5": (225, 25425),
}
ALPHAS = (0.25, 0.50, 0.75)
SOURCE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html"


@dataclass
class Portfolio:
    name: str
    r: np.ndarray        # mean returns, shape (n,)
    s: np.ndarray        # standard deviations, shape (n,)
    C: np.ndarray        # correlation matrix, shape (n, n)
    cov: np.ndarray      # covariance matrix, shape (n, n)

    @property
    def n(self):
        return len(self.r)

    @property
    def r_eq(self):
        """Return of the equal-weight portfolio u_i = 1/n."""
        return float(self.r.mean())

    @property
    def v_eq(self):
        u = np.full(self.n, 1.0 / self.n)
        return float(u @ self.cov @ u)

    @property
    def r_max(self):
        return float(self.r.max())

    @property
    def j_max(self):
        """Index of an asset attaining r_max (first one if tied)."""
        return int(np.argmax(self.r))

    def r_min(self, alpha):
        return (1.0 - alpha) * self.r_eq + alpha * self.r_max


def load_portfolio(name, data_dir=DATA_DIR):
    """Parse portN.txt, validate it against the spec, and build the covariance."""
    path = Path(data_dir) / f"{name}.txt"
    tokens = path.read_text().split()
    n = int(tokens[0])
    pos = 1
    stats = np.array(tokens[pos:pos + 2 * n], dtype=float).reshape(n, 2)
    pos += 2 * n
    rest = tokens[pos:]
    if len(rest) % 3 != 0:
        raise ValueError(f"{name}: correlation section is not a multiple of 3 tokens")
    recs = np.array(rest, dtype=float).reshape(-1, 3)

    exp_n, exp_recs = EXPECTED.get(name, (n, n * (n + 1) // 2))
    if n != exp_n:
        raise ValueError(f"{name}: expected {exp_n} assets, found {n}")
    if len(recs) != exp_recs:
        raise ValueError(f"{name}: expected {exp_recs} correlation records, found {len(recs)}")

    C = np.full((n, n), np.nan)
    seen = set()
    for i, j, c in recs:
        i, j = int(i) - 1, int(j) - 1          # 1-based in file -> 0-based arrays
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"{name}: index out of range ({i + 1}, {j + 1})")
        key = (min(i, j), max(i, j))
        if key in seen:
            raise ValueError(f"{name}: duplicate correlation record {key}")
        seen.add(key)
        C[i, j] = C[j, i] = c                  # store in both C_ij and C_ji

    if np.isnan(C).any():
        raise ValueError(f"{name}: correlation matrix has missing entries")
    if not np.allclose(np.diag(C), 1.0):
        raise ValueError(f"{name}: correlation diagonal is not 1")
    if np.abs(C).max() > 1.0 + 1e-12:
        raise ValueError(f"{name}: correlation outside [-1, 1]")

    r, s = stats[:, 0], stats[:, 1]
    if (s < 0).any():
        raise ValueError(f"{name}: negative standard deviation")
    cov = np.outer(s, s) * C
    return Portfolio(name=name, r=r, s=s, C=C, cov=cov)


def case_id(name, alpha):
    """e.g. ('port1', 0.5) -> 'port1_alpha050'."""
    return f"{name}_alpha{int(round(alpha * 100)):03d}"


def parse_case(cid):
    name, a = cid.split("_alpha")
    return name, int(a) / 100.0


def all_cases():
    return [case_id(p, a) for p in EXPECTED for a in ALPHAS]
