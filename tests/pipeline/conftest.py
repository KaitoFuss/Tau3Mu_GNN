from pathlib import Path

import tau3mu_gnn

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fail fast if the editable install (main checkout) is imported instead of this tree.
_imported_from = Path(tau3mu_gnn.__file__).resolve().parents[2]
if _imported_from != REPO_ROOT:
    raise RuntimeError(
        f"tau3mu_gnn imported from {_imported_from}, expected {REPO_ROOT}. "
        f"Run with PYTHONPATH={REPO_ROOT / 'src'}"
    )
