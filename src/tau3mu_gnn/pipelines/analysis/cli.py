"""tau3mu-pipeline (analysis): submit the SLURM jobs of all phases, one phase or one stage.

  tau3mu-pipeline all     --config my.yaml [--dry-run]
  tau3mu-pipeline phase-a --config my.yaml [--dry-run]
  tau3mu-pipeline phase-b --config my.yaml [--from phase_a_vsN] [--dry-run]
  tau3mu-pipeline phase-c --config my.yaml [--from phase_b_vsN] [--dry-run]
  tau3mu-pipeline stage <skim|bdt|gnn|eval|limit|summary> --config my.yaml
        [--from <phase>_vsN] [--cut X] [--sample signal|background] [--dry-run]

tau3mu-pipeline is installed by `pip install -e .`; `python -m tau3mu_gnn.pipelines.analysis.cli` is the same.
The jobs use paths.conda_env from the config, else the conda env that is active when you submit.
"""
from __future__ import annotations

import argparse
import logging
import sys

from tau3mu_gnn.pipelines.common.config import ConfigError

from tau3mu_gnn.pipelines.analysis.config import load_config
from tau3mu_gnn.pipelines.analysis.driver import STAGES, submit_all, submit_phase, submit_stage
from tau3mu_gnn.pipelines.common.manifest import UpstreamError
from tau3mu_gnn.pipelines.common.slurm import SubmitError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tau3mu-pipeline", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("all", "phase-a", "phase-b", "phase-c", "stage"):
        s = sub.add_parser(name, help="phases A, B and C at once" if name == "all" else None)
        if name == "stage":
            s.add_argument("stage", choices=STAGES)
            s.add_argument("--cut", type=float, help="GNN score cut (stage limit)")
            s.add_argument("--sample", choices=("signal", "background"), help="sample (stage gnn)")
        s.add_argument("--config", required=True, help="your user config file")
        if name not in ("all", "phase-a"):
            s.add_argument("--from", dest="from_", metavar="VERSION", help="upstream version, e.g. phase_a_vs2")
        s.add_argument("--dry-run", action="store_true", help="write the job scripts and print the sbatch "
                                                              "commands; submit nothing")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(a.config)
        if a.command == "all":
            subs = submit_all(cfg, dry_run=a.dry_run)
        elif a.command == "stage":
            subs = [submit_stage(cfg, a.stage, a.from_, a.cut, a.sample, dry_run=a.dry_run)]
        else:
            subs = [submit_phase(cfg, a.command.replace("-", "_"), getattr(a, "from_", None), dry_run=a.dry_run)]
    except (ConfigError, UpstreamError, SubmitError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(("DRY RUN, nothing submitted\n" if a.dry_run else "") + "\n\n".join(
          f"results: {s.version_dir}\n" +
          (f"upstream: {s.upstream}\n" if s.upstream else "") +
          f"logs:    {s.log_folder}\n" +
          "\n".join(f"  {name:<16} {job_id}" for name, job_id in s.job_ids.items()) for s in subs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
