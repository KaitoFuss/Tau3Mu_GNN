"""Required raw input branches and the user's branch map.

Names here are the analysis names in the raw input files (before stage 1).
The user config's branch_map maps analysis name -> name in the user's files.
"""
from __future__ import annotations

_TAU = "Slimmed_TauTo3Mu_"
_PF = "Slimmed_PFCands_"

# Read by every channel.
COMMON: tuple[str, ...] = (
    # stage 1 skim cut
    *(f"{_TAU}{s}" for s in ("dZmu12", "dZmu13", "dZmu23", "CosAlpha2D_LxyP3mu")),
    # tau candidate: choice, cone centre, mass fit
    *(f"{_TAU}fitted_{s}" for s in ("pt", "eta", "phi", "mass")),
    # muon directions: GNN cone and delete_muons
    *(f"{_TAU}mu{i}_{s}" for i in (1, 2, 3) for s in ("eta", "phi")),
    # GNN node features
    *(f"{_PF}{s}" for s in ("pt", "eta", "phi", "mass", "pdgId", "charge", "puppiWeight", "d0", "dz")),
    # event id: merging scores back into the trees
    "run", "luminosityBlock", "event",
)

# BDT ISO features and GNN graph-level features differ per channel.
PER_CHANNEL: dict[str, tuple[str, ...]] = {
    "W": (
        *(f"{_TAU}{s}" for s in ("sigLxy_3muVtxBS", "vtx_chi2", "vtx_Ndof",
                                 "iso_ptChargedFromPV", "iso_ptPhotons", "iso_ptChargedFromPU")),
        *(f"TauPlusMET_{s}" for s in ("Tau_Puppi_mT", "PuppiMET_pt", "Puppi_pt", "Puppi_phi",
                                      "PuppiMETminPz", "PuppiMETmaxPz")),
    ),
}


class BranchError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Input branch problems:\n" + "\n".join(f"  - {p}" for p in self.problems))


def required_branches(channel: str) -> tuple[str, ...]:
    if channel not in PER_CHANNEL:
        raise BranchError([f"no required-branch list for channel {channel!r} (known: {sorted(PER_CHANNEL)})"])
    return COMMON + PER_CHANNEL[channel]
