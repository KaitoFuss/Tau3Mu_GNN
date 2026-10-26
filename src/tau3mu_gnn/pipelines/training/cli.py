"""tau3mu training driver: submit the SLURM jobs of all training phases or one phase.

  tau3mu-training all       --config my_training.yaml [--dry-run]
  tau3mu-training bdt       --config my_training.yaml [--dry-run]
  tau3mu-training gnn       --config my_training.yaml [--from bdt_vsN] [--dry-run]
  tau3mu-training benchmark --config my_training.yaml [--from bdt_vsN] [--dry-run]

tau3mu-training is installed by `pip install -e .`; `python -m tau3mu_gnn.pipelines.training.cli` is the same.
The jobs use paths.conda_env from the config, else the conda env that is active when you submit.
"""
from __future__ import annotations

import argparse
import logging
import sys

from tau3mu_gnn.pipelines.common.config import ConfigError
from tau3mu_gnn.pipelines.common.manifest import UpstreamError
from tau3mu_gnn.pipelines.common.slurm import SubmitError
from tau3mu_gnn.pipelines.training.config import load_training_config
from tau3mu_gnn.pipelines.training.driver import submit_all, submit_phase


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tau3mu-training", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("all", "bdt", "gnn", "benchmark"):
        s = sub.add_parser(name, help="bdt, then gnn and benchmark" if name == "all" else None)
        s.add_argument("--config", required=True, help="your training config file")
        if name in ("gnn", "benchmark"):
            s.add_argument("--from", dest="from_", metavar="VERSION",
                           help="bdt version, e.g. bdt_vs2 (default: the highest complete one)")
        s.add_argument("--dry-run", action="store_true", help="write the job scripts and print the sbatch "
                                                              "commands; submit nothing")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_training_config(a.config)
        if a.command == "all":
            subs = submit_all(cfg, dry_run=a.dry_run)
        else:
            subs = [submit_phase(cfg, a.command, getattr(a, "from_", None), dry_run=a.dry_run)]
    except (ConfigError, UpstreamError, SubmitError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(("DRY RUN, nothing submitted\n" if a.dry_run else "") + "\n\n".join(
          f"results: {s.version_dir}\n" +
          (f"upstream: {s.upstream}\n" if s.upstream else "") +
          f"logs:    {s.log_folder}\n" +
          "\n".join(f"  {name:<22} {job_id}" for name, job_id in s.job_ids.items()) for s in subs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
