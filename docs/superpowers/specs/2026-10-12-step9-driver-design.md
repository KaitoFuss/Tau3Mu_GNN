# Step 9 — Driver: Design

Status: agreed with Kaito in chat, 2026-10-12. Implemented in two mini steps: **9a** (no SLURM) and **9b** (SLURM).
Parent document: `docs/wrapper_roadmap.md` (sec. 5 design, sec. 6 step 9). Where this spec differs from the roadmap, this spec wins and the roadmap gets updated (sec. 9).

## 1. Goal

A user with one config file runs three phase commands, one after the other. Each command submits the SLURM jobs of its phase and returns. At the end of phase C the run folder holds the expected limit table (`limits.json`).

```
python -m tau3mu_gnn.pipeline.cli phase-a --config my.yaml [--dry-run]
python -m tau3mu_gnn.pipeline.cli phase-b --config my.yaml [--from phase_a_vsN] [--dry-run]
python -m tau3mu_gnn.pipeline.cli phase-c --config my.yaml [--from phase_b_vsN] [--dry-run]
python -m tau3mu_gnn.pipeline.cli stage <skim|bdt|gnn|eval|limit|summary> --config my.yaml [--cut X] [--sample S] [--from ...] [--dry-run]
```

The bash wrapper `tau3mu-run` (env activation) is step 10. The `all` command is step 10b.

## 2. Decisions (Kaito, 2026-10-12)

| # | Decision |
|---|---|
| D1 | Split: 9a = run folders, versions, manifest, checks, storage helper, job functions (no SLURM, tested with toy files). 9b = job scripts, `sbatch`, proxy check, CLI. |
| D2 | Storage: `output_base` may be `root://<host>:<port>//pnfs/...` (dCache) or a local path. dCache: **read** through the `/pnfs/...` NFS path, **write** with `xrdcp` / `xrdfs`. Local: plain file copy. |
| D3 | **No local phase runs.** Phase commands and `stage` only submit SLURM jobs. Job bodies are plain Python functions; tests call them directly. |
| D4 | **Every phase run makes a new version of that phase**: `phase_a_vs1`, `phase_b_vs1`, `phase_b_vs2`, ... Nothing is overwritten. (Supersedes roadmap: "run folder `<analysis_name>_vsN`" and "cut scans add `cut_<value>/` inside the same run".) |
| D5 | Single-stage runs go to `debug/<stage>_vsN/`. No phase ever reads from `debug/`. |
| D6 | Logs: new required user key `paths.log_dir` (local disk, e.g. `/work/$USER/tau3mu_logs`). Job scripts and SLURM `.out`/`.err` go there. Each job also copies its log next to the results (`<phase>_vsN/logs/`). |
| D7 | SLURM resources live in the **frozen** `analysis_defaults_<channel>.yaml` (`slurm:` section). No `slurm:` in the user file. (Supersedes roadmap sec. 5.1 / decision log 2026-10-07: `slurm` in the user file.) |
| D8 | Job submission: Python writes one bash script per job and submits it with `sbatch --parsable [--dependency=...]` (approach A). No new dependency. |
| D9 | 9b real-cluster check: a small-input config (subsets of Kaito's W files), phases A → B → C on T3. Kaito queues it. The full W run vs the reference limit stays in step 11. |

## 3. Layout

```
<output_base>/<analysis_name>/                  results (dCache or local)
├── phase_a_vs1/
│   ├── manifest.json
│   ├── config_used.yaml                        user file + defaults as loaded (merged view)
│   ├── 0_resonance_skim/                       only if stage 0 copies (non-empty branch map)
│   ├── 1_skim/{signal,background}.root
│   ├── 2_bdt/...                               stage 2 outputs (ROOT + pickle)
│   └── logs/<job>-<jobid>.{out,err}
├── phase_b_vs1/
│   ├── manifest.json, config_used.yaml, logs/
│   ├── 3_gnn/{signal,background}_with_gnn_scores.root, raw_scores_and_tau_mass_{signal,background}.npz
│   └── 4_eval/*.png
├── phase_c_vs1/
│   ├── manifest.json, config_used.yaml, logs/
│   ├── cut_<cut>/...                           stage 5 per cut (step 8)
│   └── limits.{json,txt,png}                   summarize (step 8b)
└── debug/<stage>_vsN/                          single-stage runs (D5), config_used.yaml + logs/, no manifest

<paths.log_dir>/<analysis_name>/<phase or debug stage>_vsN/
├── <job>.sh                                    generated job scripts
└── <job>-<jobid>.{out,err}                     SLURM logs
```

Stage folder names inside a phase are as above (`1_skim`, `2_bdt`, `3_gnn`, `4_eval`). Phase C has no stage subfolder (stage 5 writes `cut_<cut>/` directly into the phase folder, as in step 8b).

## 4. Versions and upstream choice

- `next_version(base, prefix)`: lists `<base>` (local path; for dCache the `/pnfs` path), finds `<prefix>_vs<N>` entries, returns `<prefix>_vs<max N + 1>` (`_vs1` if none). Same idea as `TrainingUtils.generate_name_with_versionnumber`.
- The submit command creates the folder (local `mkdir` / `xrdfs mkdir -p`) and writes `config_used.yaml` there **before** submitting. So two quick submits get different versions.
- **Complete** version = its `manifest.json` exists.
- Default upstream for phase B / C and for `stage`: the **highest complete** version of the phase before it. If a newer incomplete version exists, print a note (`phase_a_vs3 has no manifest (failed or still running); using phase_a_vs2`).
- `--from <phase>_vsN` picks a version. It must be complete, else error.
- Upstream of each stage for `stage <name>`: `skim` ← the config inputs (like phase A); `bdt` ← phase A (`1_skim/`); `gnn`, `eval` ← phase A / phase B; `limit`, `summary` ← phase B / phase C. Exact table in sec. 6.

## 5. Manifest

`<phase>_vsN/manifest.json`, written by the last job of the phase, only when all jobs before it succeeded (`afterok`), except phase C (sec. 6).

```json
{
  "phase": "phase_b",
  "version": "phase_b_vs2",
  "upstream": "phase_a_vs1",
  "outputs": {"signal": "3_gnn/signal_with_gnn_scores.root", "...": "..."},
  "settings": {"inputs": {...}, "branch_map": {...}, "skim": {...}, "bdt": {...}, "gnn": {...}},
  "missing": [],
  "git_commit": "7aeeaad",
  "created": "2026-10-12T18:00:00+02:00",
  "slurm_jobs": {"gnn_signal": "123", "gnn_background": "124", "eval": "125"}
}
```

- `outputs`: paths **relative to the phase folder**.
- `settings`: the config sections that affect this phase **and every phase before it**:
  - phase A: `channel`, `inputs`, `branch_map`, `skim`, `bdt`
  - phase B: phase A keys + `gnn`
  - phase C: phase B keys + `limit`
- `missing`: phase C only, the cuts without a limit (from `summarize`).
- **Checks in the first job of phase B / C** (`check_upstream`):
  - upstream manifest missing → error (`UpstreamError`), job fails, later jobs do not start;
  - an `outputs` file missing → error, listing all missing files;
  - `settings` of the upstream phases differ from the current config → **warning** with the differing keys (dotted names, old → new). The job goes on.
- The submit command does a quick version of the first two checks (upstream version exists and is complete), so obvious mistakes fail before `sbatch`.

## 6. Jobs

| Phase | Job | Stage(s) | Inputs | Writes (in phase folder) | Depends on |
|---|---|---|---|---|---|
| A | `skim` | 0 + 1 | config inputs | `0_resonance_skim/`, `1_skim/` | — |
| A | `bdt` | 2 | `1_skim/` (same phase) | `2_bdt/`, **manifest** | `afterok:skim` |
| B | `gnn_signal` | 3, signal | phase A `2_bdt/` | `3_gnn/signal_*` | — (runs `check_upstream`) |
| B | `gnn_background` | 3, background | phase A `2_bdt/` | `3_gnn/background_*` | — (runs `check_upstream`) |
| B | `eval` | 4 | `3_gnn/` (same phase) + phase A pickle | `4_eval/`, **manifest** | `afterok:gnn_signal:gnn_background` |
| C | `limit_<cut>` | 5, one cut | phase B `3_gnn/` | `cut_<cut>/` | — (runs `check_upstream`) |
| C | `summary` | `summarize` | `cut_*/limit.json` (same phase) | `limits.*`, **manifest** (with `missing`) | `afterany:all limit jobs` |

- Both phase B GNN jobs and every phase C limit job run `check_upstream` (they all start at once; each must stop if the upstream is bad).
- Stage 3 gets a `samples` argument (`run(cfg, inputs, out_dir, samples=SAMPLES)`), so signal and background run as two jobs.
- Phase C summary: if `summarize` raises (no cut has a limit), no manifest → phase C version incomplete.
- `stage <name>`: one job with the same body as the phase job(s) of that stage, output in `debug/<stage>_vsN/`, no manifest. `stage gnn --sample <signal|background>` scores one sample (required argument). `stage limit --cut X` (X must be in `limit.gnn_score_cut`). `stage summary --from phase_c_vsN` re-summarizes the `cut_*/` of that phase C version into the debug folder.

**Job body** (`pipeline/jobs.py`, one function per job, signature `(cfg, phase_dir, upstream_dir, **job_args)`):
1. first job(s) of B / C: `check_upstream`;
2. resolve inputs to readable paths (dCache → `/pnfs`);
3. run the stage with `out_dir` = local scratch (`$TMPDIR/<stage folder>`);
4. `storage.put(scratch_dir, phase_dir/<stage folder>)` (xrdcp or copy);
5. manifest job: `write_manifest(...)`.

## 7. Job script and submission (9b)

One bash template, filled per job:

```bash
#!/bin/bash
#SBATCH --job-name=<analysis_name>.<phase>_vsN.<job>
#SBATCH --account=<slurm.account>  -p <partition>  --mem=<mem>  --time=<time>  --cpus-per-task=<cpus>
#SBATCH -o <log_dir>/<analysis_name>/<phase>_vsN/<job>-%j.out
#SBATCH -e <log_dir>/<analysis_name>/<phase>_vsN/<job>-%j.err
set -euo pipefail
JOB_SCRATCH=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$JOB_SCRATCH"; export TMPDIR="$JOB_SCRATCH"
cleanup() { <copy own .out/.err to <phase>_vsN/logs/ (best effort)>; rm -rf "$JOB_SCRATCH"; }
trap cleanup EXIT
cd "$JOB_SCRATCH"
<python> -m tau3mu_gnn.pipeline.jobs <job> --config <abs user file> --phase-dir <phase dir> [--upstream <dir>] [--cut X]
```

- `<python>` = absolute path of the Python that ran the submit command (`sys.executable`). The env is not activated in the job; `combine` is found next to that Python (step 8). `paths.conda_env` stays a step 10 topic.
- The log copy in `cleanup` misses the last lines written after it; accepted.
- `slurm.py`: `submit(script, dependency=None) -> job_id` runs `sbatch --parsable`; dependency strings `afterok:1:2`, `afterany:...`. `--dry-run`: write scripts, print the `sbatch` commands with placeholder IDs, submit nothing.
- **Proxy check** at submit time (only when `output_base` is dCache): `X509_USER_PROXY` set and the file is not under `/tmp` (nodes cannot see it); `voms-proxy-info --exists --valid <H>:00` with H = longest `time` of the phase's jobs, in hours. On failure: error with `voms-proxy-init --voms cms --valid 192:00`. (With the sec. 8 times, phase B needs ≥ 48 h. If times are raised, the check follows.)
- After submit: print phase folder, log folder and job IDs.

## 8. Config changes

- User file: `paths.log_dir` — **required**, local path, `$USER` / `~` expanded (as `output_base`). Error if it starts with `root://` or `/pnfs`.
- Frozen `analysis_defaults_W.yaml`, new section (values from the old job scripts; `time` where the old script had none, and eval / summary memory, are first guesses):

```yaml
slurm:
  account: t3
  jobs:
    skim:           {partition: standard, mem: 128G, time: "0-03:00:00", cpus: 4}
    bdt:            {partition: standard, mem: 64G,  time: "0-06:00:00", cpus: 1}
    gnn_signal:     {partition: long,     mem: 64G,  time: "1-00:00:00", cpus: 1}
    gnn_background: {partition: long,     mem: 64G,  time: "2-00:00:00", cpus: 1}
    eval:           {partition: standard, mem: 32G,  time: "0-01:00:00", cpus: 1}
    limit:          {partition: standard, mem: 128G, time: "0-01:30:00", cpus: 1}
    summary:        {partition: short,    mem: 4G,   time: "0-00:30:00", cpus: 1}
```

- `stage gnn --sample <signal|background>` uses the resources of `gnn_signal` / `gnn_background`.
- Times are first values (Kaito, 2026-10-12); raise them in this file if jobs hit the limit.
- `Config.output_dir` stays `<output_base>/<analysis_name>`; phase folders are below it.
- `configs/example_W.yaml`: add `paths.log_dir`.
- The `slurm:` section is read in 9a (config) but used only in 9b.

## 9. Roadmap updates

- Sec. 5 layout → sec. 3 of this spec. Sec. 5.1: `slurm` moves to the defaults file; `paths.log_dir` added. Sec. 5.2: add `jobs.py`.
- Step 9 line → split into 9a / 9b with a pointer to this spec. Step 8b note "summary job with `afterany`" stays.
- Decision log rows for D1–D9.

## 10. Error handling (summary)

| Situation | When | Result |
|---|---|---|
| Bad config (missing / unknown key, bad value) | submit | error, nothing submitted |
| No complete upstream version / `--from` incomplete | submit | error, nothing submitted |
| Proxy missing, under `/tmp`, or too short | submit (dCache only) | error with `voms-proxy-init` command |
| `sbatch` fails | submit | error; jobs already submitted in this call are listed so the user can `scancel` them |
| Upstream manifest or file missing | first job(s) | job fails, later jobs do not start (`afterok`) |
| Upstream settings differ | first job(s) | warning in the job log, job goes on |
| Stage error (any) | job | job fails; no manifest (A, B); phase C: missing row, summary still runs |
| `xrdcp` fails | job | job fails (non-zero exit) |

## 11. Testing

**9a** (pytest, login node, toy ROOT files, local `output_base` = `tmp_path`):
- versions: `next_version`, highest complete, note on newer incomplete, `--from` incomplete → error;
- storage: `root://host:port//pnfs/x` ↔ `/pnfs/x`; local copy; `xrdcp` / `xrdfs` command lines built right (fake executables on `PATH`);
- manifest: write / read round trip; `check_upstream` missing manifest, missing file, settings diff warning (names the keys);
- job functions: phase A → B → C on toy files with stages faked where they need real models / Combine (the stages themselves are tested in steps 4–8); the end-to-end chain with real stages where the toy fixtures already exist;
- stage 3 `samples` argument; config `paths.log_dir`, `slurm` section.

**9b**:
- fake `sbatch` on `PATH` (records arguments, prints increasing IDs): dependency strings, script content (`#SBATCH` lines, python path, arguments), `--dry-run` submits nothing;
- proxy check with fake `voms-proxy-info`;
- real T3 check (D9): small-input config, A → B → C, Kaito queues. Pass = all jobs succeed, three complete versions with manifests, `limits.json` written, logs copied, a second `phase-b` gives `phase_b_vs2` and `phase-c` then uses it.

## 12. Out of scope

`all` command (10b), bash wrapper + README (10), HTCondor (13), retries / resubmit of failed jobs, cleanup of old versions, `max_events` in the config.
