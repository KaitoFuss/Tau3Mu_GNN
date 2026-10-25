"""Stage 0: branch map, then resonance skim (raw NanoAOD -> *_skimmed_no_resonances.root).

Order: check inputs -> rename (branch map) -> resonance skim -> write, in one RDataFrame pass.
The resonance skim therefore works on analysis names. Resonance-skimmed inputs with an empty
branch map are passed on, no copy.

Resonance skim (from Riccardo's tau3mu_gnn_pf_skimmer.py, plus the mass cut and the veto):
a tau candidate passes if vertex chi2 < vtx_chi2_max, low < m(3 mu) < high, and no opposite-sign
dimuon mass is within veto_half_width of a veto mass. Masses are 4-vector sums of the muons
(double precision, muon mass MUON_MASS). Events without a passing candidate are removed. TauTo3Mu_* -> Slimmed_TauTo3Mu_*
(passing candidates); TauPlusMET_* (one entry per candidate) are masked in place, so they stay
aligned; PFCands_* -> Slimmed_PFCands_* (DeltaR < pf_dr_max to any passing candidate).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from tau3mu_gnn.pipelines.common.branches import BranchError, check_input_branches, rename_columns
from tau3mu_gnn.pipelines.common.config import ResonanceSkimConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type hints only: common code does not import the analysis package at run time
    from tau3mu_gnn.pipelines.analysis.config import Config

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
PAIRS = ((1, 2), (1, 3), (2, 3))
MUON_MASS = 0.10566  # GeV

_CPP = """
#ifndef TAU3MU_RESONANCE_SKIM
#define TAU3MU_RESONANCE_SKIM
#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"

namespace tau3mu_rs {
using ROOT::VecOps::RVec;
using P4 = ROOT::Math::PtEtaPhiMVector;  // double precision

template <typename T>
RVec<double> mass2(const RVec<T>& pt1, const RVec<T>& eta1, const RVec<T>& phi1, const RVec<T>& m1,
                   const RVec<T>& pt2, const RVec<T>& eta2, const RVec<T>& phi2, const RVec<T>& m2) {
    RVec<double> out(pt1.size());
    for (size_t i = 0; i < pt1.size(); ++i)
        out[i] = (P4(pt1[i], eta1[i], phi1[i], m1[i]) + P4(pt2[i], eta2[i], phi2[i], m2[i])).M();
    return out;
}

template <typename T>
RVec<double> mass3(const RVec<T>& pt1, const RVec<T>& eta1, const RVec<T>& phi1, const RVec<T>& m1,
                   const RVec<T>& pt2, const RVec<T>& eta2, const RVec<T>& phi2, const RVec<T>& m2,
                   const RVec<T>& pt3, const RVec<T>& eta3, const RVec<T>& phi3, const RVec<T>& m3) {
    RVec<double> out(pt1.size());
    for (size_t i = 0; i < pt1.size(); ++i)
        out[i] = (P4(pt1[i], eta1[i], phi1[i], m1[i]) + P4(pt2[i], eta2[i], phi2[i], m2[i])
                  + P4(pt3[i], eta3[i], phi3[i], m3[i])).M();
    return out;
}

// Per element of collection 1: minimum DeltaR to any element of collection 2 (999 if it is empty).
template <typename T1, typename T2>
RVec<float> min_dr(const RVec<T1>& eta1, const RVec<T1>& phi1, const RVec<T2>& eta2, const RVec<T2>& phi2) {
    RVec<float> out(eta1.size(), 999.f);
    for (size_t i = 0; i < eta1.size(); ++i)
        for (size_t j = 0; j < eta2.size(); ++j) {
            float dr = ROOT::VecOps::DeltaR<float>(eta1[i], eta2[j], phi1[i], phi2[j]);
            if (dr < out[i]) out[i] = dr;
        }
    return out;
}
}  // namespace tau3mu_rs
#endif
"""


def _kin(i: int) -> str:
    return ", ".join(f"TauTo3Mu_mu{i}_{s}" for s in ("pt", "eta", "phi", "mass"))


def _resonance_skim(df, columns: list[str], rs: ResonanceSkimConfig):
    """Apply the resonance skim to df (raw analysis names). Returns (df, output columns)."""
    import ROOT

    ROOT.gInterpreter.Declare(_CPP)
    columns = list(columns)
    for i in (1, 2, 3):  # raw NanoAOD has no muon mass branch; the old skims used this constant
        name = f"TauTo3Mu_mu{i}_mass"
        if name not in columns:
            df = df.Define(name, f"ROOT::RVecF(TauTo3Mu_mu{i}_pt.size(), {MUON_MASS!r}f)")
            columns.append(name)
    lo, hi = rs.mass_range
    df = df.Define("_rs_m123", f"tau3mu_rs::mass3({_kin(1)}, {_kin(2)}, {_kin(3)})")
    masks = ["TauTo3Mu_pass_chi2_mask", "TauTo3Mu_pass_mass_mask"]
    df = df.Define(masks[0], f"TauTo3Mu_vtx_chi2 < {rs.vtx_chi2_max!r}")
    df = df.Define(masks[1], f"_rs_m123 > {lo!r} && _rs_m123 < {hi!r}")
    for i, j in PAIRS:
        m = f"_rs_m{i}{j}"
        near = " || ".join(f"abs({m} - {v!r}) < {rs.veto_half_width!r}" for v in rs.veto_masses)
        df = df.Define(m, f"tau3mu_rs::mass2({_kin(i)}, {_kin(j)})")
        masks.append(f"TauTo3Mu_pass_m{i}{j}_mask")
        df = df.Define(masks[-1], f"!(TauTo3Mu_mu{i}_charge * TauTo3Mu_mu{j}_charge < 0 && ({near}))")
    df = df.Define("TauTo3Mu_pass_mask", " && ".join(masks))
    df = df.Filter("Sum(TauTo3Mu_pass_mask) > 0", "stage 0: at least one tau candidate passes")

    tau = [c for c in columns if c.startswith("TauTo3Mu_")] + masks + ["TauTo3Mu_pass_mask"]
    for c in tau:
        df = df.Define(f"Slimmed_{c}", f"{c}[TauTo3Mu_pass_mask]")
    df = df.Define("nSlimmedTauTo3Mu", "static_cast<UInt_t>(Slimmed_TauTo3Mu_fitted_pt.size())")
    met = [c for c in columns if c.startswith("TauPlusMET_")]
    for c in met:
        df = df.Redefine(c, f"{c}[TauTo3Mu_pass_mask]")
    if "nTauPlusMET" in columns:
        df = df.Redefine("nTauPlusMET", f"static_cast<{df.GetColumnType('nTauPlusMET')}>(Sum(TauTo3Mu_pass_mask))")

    df = df.Define("PFCands_pass_dr_mask",
                   "tau3mu_rs::min_dr(PFCands_eta, PFCands_phi, Slimmed_TauTo3Mu_fitted_eta, Slimmed_TauTo3Mu_fitted_phi)"
                   f" < {rs.pf_dr_max!r}")
    pf = [c for c in columns if c.startswith("PFCands_")] + ["PFCands_pass_dr_mask"]
    for c in pf:
        df = df.Define(f"Slimmed_{c}", f"{c}[PFCands_pass_dr_mask]")
    df = df.Define("nSlimmedPFCands", "static_cast<UInt_t>(Slimmed_PFCands_pt.size())")

    out = [c for c in columns if not c.startswith(("TauTo3Mu_", "PFCands_"))]
    out += [f"Slimmed_{c}" for c in tau] + ["nSlimmedTauTo3Mu"] + [f"Slimmed_{c}" for c in pf] + ["nSlimmedPFCands"]
    return df, out


def check_tauplusmet_aligned(frames) -> None:
    """Raise BranchError if, in any event of resonance-skimmed inputs, TauPlusMET_* has not one entry per
    Slimmed_TauTo3Mu candidate. Old resonance skims slimmed TauTo3Mu_* but not TauPlusMET_*, so the entry of a
    candidate cannot be found (E1). frames: sample -> (RDataFrame, columns), branch map applied."""
    problems = []
    for sample, (df, columns) in frames.items():
        met = sorted(c for c in columns if c.startswith("TauPlusMET_"))
        if not met or "Slimmed_TauTo3Mu_fitted_pt" not in columns:
            continue
        if "RVec" not in df.GetColumnType(met[0]):  # flat branch (toy test inputs): no candidates to align
            continue
        bad = df.Filter(f"{met[0]}.size() != Slimmed_TauTo3Mu_fitted_pt.size()").Count()
        total = df.Count()
        if bad.GetValue():
            problems.append(f"{sample}: {bad.GetValue()} of {total.GetValue()} events have len({met[0]}) != "
                            "len(Slimmed_TauTo3Mu_fitted_pt): TauPlusMET_* was not slimmed with TauTo3Mu_* (old "
                            "resonance skim). Skim from raw NanoAOD instead (already_resonance_skimmed: false)")
    if problems:
        raise BranchError(problems)


def run(cfg: Config, out_dir: str | Path, extra_branches: Sequence[str] = ()) -> dict[str, list[str]]:
    """Return the input files of stage 1 per sample. Checks all inputs before any work (resonance-skimmed
    inputs: also check_tauplusmet_aligned).
    extra_branches: further required input branches (training: branches.TRAINING_EXTRA)."""
    inputs = cfg.inputs
    raw = not inputs.already_resonance_skimmed
    files = {sample: check_input_branches(getattr(inputs, sample), inputs.tree_name, cfg.branch_map, cfg.channel,
                                          raw=raw, extra=extra_branches)
             for sample in SAMPLES}

    import ROOT

    ROOT.EnableImplicitMT()
    frames = {}
    for sample in SAMPLES:
        frames[sample] = rename_columns(ROOT.RDataFrame(inputs.tree_name, files[sample]), cfg.branch_map)
    if not raw:
        check_tauplusmet_aligned(frames)
        if all(user == analysis for analysis, user in cfg.branch_map.items()):
            log.info("stage 0: inputs already resonance-skimmed, empty branch map: inputs passed on, no copy")
            return files
    else:
        frames = {sample: _resonance_skim(df, columns, cfg.resonance_skim) for sample, (df, columns) in frames.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for sample, (df, columns) in frames.items():
        output = out_dir / f"{sample}.root"
        log.info("stage 0: %s (%d file(s)) -> %s", sample, len(files[sample]), output)
        df.Snapshot(inputs.tree_name, str(output), columns)
        outputs[sample] = [str(output)]
    return outputs
