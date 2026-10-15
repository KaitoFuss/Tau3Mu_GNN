"""tau3mu pipeline driver: submit the SLURM jobs of one phase or one stage.

  python -m tau3mu_gnn.pipeline.cli phase-a --config my.yaml [--dry-run]
  python -m tau3mu_gnn.pipeline.cli phase-b --config my.yaml [--from phase_a_vsN] [--dry-run]
  python -m tau3mu_gnn.pipeline.cli phase-c --config my.yaml [--from phase_b_vsN] [--dry-run]
  python -m tau3mu_gnn.pipeline.cli stage <skim|bdt|gnn|eval|limit|summary> --config my.yaml
        [--from <phase>_vsN] [--cut X] [--sample signal|background] [--dry-run]

Run it in the activated conda env (conda activate Tau3Mu); the jobs use the same env.
"""
from __future__ import annotations

import argparse
import logging
import sys

from tau3mu_gnn.pipeline.config import ConfigError, load_config
from tau3mu_gnn.pipeline.driver import STAGES, submit_phase, submit_stage
from tau3mu_gnn.pipeline.manifest import UpstreamError
from tau3mu_gnn.pipeline.slurm import SubmitError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tau3mu_gnn.pipeline.cli", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("phase-a", "phase-b", "phase-c", "stage"):
        s = sub.add_parser(name)
        if name == "stage":
            s.add_argument("stage", choices=STAGES)
            s.add_argument("--cut", type=float, help="GNN score cut (stage limit)")
            s.add_argument("--sample", choices=("signal", "background"), help="sample (stage gnn)")
        s.add_argument("--config", required=True, help="your user config file")
        if name != "phase-a":
            s.add_argument("--from", dest="from_", metavar="VERSION", help="upstream version, e.g. phase_a_vs2")
        s.add_argument("--dry-run", action="store_true", help="write the job scripts and print the sbatch "
                                                              "commands; submit nothing")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(a.config)
        if a.command == "stage":
            s = submit_stage(cfg, a.stage, a.from_, a.cut, a.sample, dry_run=a.dry_run)
        else:
            s = submit_phase(cfg, a.command.replace("-", "_"), getattr(a, "from_", None), dry_run=a.dry_run)
    except (ConfigError, UpstreamError, SubmitError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(("DRY RUN, nothing submitted\n" if a.dry_run else "") +
          f"results: {s.version_dir}\n" +
          (f"upstream: {s.upstream}\n" if s.upstream else "") +
          f"logs:    {s.log_folder}\n" +
          "\n".join(f"  {name:<16} {job_id}" for name, job_id in s.job_ids.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
