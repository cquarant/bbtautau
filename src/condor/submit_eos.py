#!/usr/bin/env python3

"""
Batch-generate HTCondor .sub (JDL) + matching _eos.sh scripts, like submit.py,
but targeting CERN EOS instead of T2/xrdcp.

Features:
- Reads data/index_<year>.json to get per-(sample,subsample) file counts when available.
- Splits into multiple jobs using --files-per-job (or YAML overrides), generating one .sub+.sh per job.
- Mirrors original naming and args: .../<year>_<subsample>_<jobindex>_eos.{sub,sh}
- JDL uses x509 at /afs/... , Singularity + CVMFS, JobFlavour, getenv, and ClusterId-based logs.
- SH clones repo, records commithash to EOS, runs src/run.py with correct [starti, endi), copies outputs to EOS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

import yaml  # pip install pyyaml

# -----------------------------
# Templates
# -----------------------------

JDL_TEMPLATE = """\
executable              = {exe_rel}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
request_memory          = {request_memory}

use_x509userproxy       = true
x509userproxy           = {x509_path}

+SingularityImage       = "{singularity_image}"
+SingularityBindCVMFS   = True
+JobFlavour             = "{job_flavour}"

getenv                  = True

output                  = {logs_dir}/$(ClusterId).out
error                   = {logs_dir}/$(ClusterId).err
log                     = {logs_dir}/$(ClusterId).log

Queue 1
"""

SH_TEMPLATE = r"""
#!/bin/bash
set -e

# ========= 1) basic =========
export HOME=$(pwd)
export PATH="$HOME/.local/bin:$PATH"

# EOS
eos_target="root://eosuser.cern.ch//{eos_base}"

mkdir -p outfiles

# ========= 2) Clone =========
(
    r=3
    while ! git clone --single-branch --recursive --branch {git_branch} --depth=1 https://github.com/{git_user}/bbtautau
    do
        ((--r)) || exit 1
        sleep 60
        rm -rf bbtautau
        echo "Retry cloning..."
    done
)

cd bbtautau || exit 1

# 记录 commit hash
commithash=$(git rev-parse HEAD)
echo "https://github.com/{git_user}/bbtautau/commit/${{commithash}}" > commithash.txt

# 上传 hash（用 xrdcp，防止写锁）
xrdcp -f commithash.txt ${{eos_target}}/jobchecks/commithash_{job_index}.txt || true

# ========= 3) Install (写在 HOME/.local，不污染系统) =========
pip install --user -e .
cd boostedhh && pip install --user -e . && cd ..

# ========= 4) 运行 =========
python -u -W ignore src/run.py \
    --year {year} --starti {starti} --endi {endi} --batch-size {batch_size} --file-tag {job_index} \
    --samples {sample} --subsamples {subsample} --processor {processor} \
    --maxchunks {maxchunks} --chunksize {chunksize} \
    --{save_root_flag_toggle} --{save_systs_flag_toggle} \
    --nano-version {nano_version} --region {region} \
    --{bb_preselection_toggle}

# ========= 5) 上传输出（全部用 xrdcp）=========
xrdcp -f num_batches*.txt   ${{eos_target}}/jobchecks/ || true
xrdcp -f outfiles/*         ${{eos_target}}/pickles/out_{job_index}.pkl
xrdcp -f *.parquet          ${{eos_target}}/parquet/ || true
xrdcp -f *.root             ${{eos_target}}/root/    || true

# ========= 6) 清理本地 =========
rm -f *.parquet *.root *.txt
"""


# -----------------------------
# Helpers
# -----------------------------


def load_index_json(year: str) -> dict:
    path = Path(f"data/index_{year}.json")
    if not path.exists():
        return {}
    with path.open() as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def infer_nfiles(index_obj: dict, sample: str, subsample: str) -> int | None:
    """
    Try to infer total files for given (sample, subsample) from index json.
    Accepts structures like:
      {"HHbbtt": {"VBF...": ["file1.root","file2.root", ...]}}
    or
      {"HHbbtt": {"VBF...": {"files": [...] , "n_files": 123}}}
    or
      {"HHbbtt": {"VBF...": 123}}
    """
    if not index_obj:
        return None
    s = index_obj.get(sample)
    if s is None:
        return None
    sub = s.get(subsample)
    if sub is None:
        return None
    if isinstance(sub, int):
        return sub
    if isinstance(sub, list):
        return len(sub)
    if isinstance(sub, dict):
        if "n_files" in sub and isinstance(sub["n_files"], int):
            return sub["n_files"]
        if "files" in sub and isinstance(sub["files"], list):
            return len(sub["files"])
    # fallback
    try:
        return int(sub)  # last-ditch
    except Exception:
        return None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_paths(tag: str, nano_version: str, region: str, year: str, subsample: str, job_index: int):
    base_rel = Path("condor") / "skimmer" / f"{tag}_{nano_version}_{region}"
    logs_dir = base_rel / "logs"
    sh_name = f"{year}_{subsample}_{job_index}_eos.sh"
    sub_name = f"{year}_{subsample}_{job_index}_eos.sub"
    exe_rel = base_rel / sh_name
    sub_rel = base_rel / sub_name
    return base_rel, logs_dir, exe_rel, sub_rel


def write_text(path: Path, content: str, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.chmod(path, mode)


# -----------------------------
# Core generation
# -----------------------------


def render_jdl(
    sub_path: Path,
    exe_rel: Path,
    logs_dir: Path,
    x509_path: str,
    request_memory: int,
    singularity_image: str,
    job_flavour: str,
):
    content = JDL_TEMPLATE.format(
        exe_rel=str(exe_rel),
        logs_dir=str(logs_dir),
        x509_path=x509_path,
        request_memory=request_memory,
        singularity_image=singularity_image,
        job_flavour=job_flavour,
    )
    write_text(sub_path, content)


def render_sh(
    sh_path: Path,
    eos_user_path: str,
    tag: str,
    nano_version: str,
    region: str,
    year: str,
    subsample: str,
    job_index: int,
    git_user: str,
    git_branch: str,
    run_kwargs: dict,
):
    eos_base = f"/eos/user/{eos_user_path}/bbtautau/skimmer/{tag}_{nano_version}_{region}/{year}/{subsample}"

    save_root_flag_toggle = "save-root" if run_kwargs.get("save_root", False) else "no-save-root"
    save_systs_flag_toggle = (
        "save-systematics" if run_kwargs.get("save_systematics", False) else "no-save-systematics"
    )
    bb_preselection_toggle = (
        "fatjet-bb-preselection"
        if run_kwargs.get("fatjet_bb_preselection", False)
        else "no-fatjet-bb-preselection"
    )

    content = SH_TEMPLATE.format(
        eos_base=eos_base,
        git_user=git_user,
        git_branch=git_branch,
        year=year,
        starti=run_kwargs["starti"],
        endi=run_kwargs["endi"],
        batch_size=run_kwargs["batch_size"],
        job_index=job_index,
        sample=run_kwargs["sample"],
        subsample=run_kwargs["subsample"],
        processor=run_kwargs["processor"],
        maxchunks=run_kwargs["maxchunks"],
        chunksize=run_kwargs["chunksize"],
        nano_version=run_kwargs["nano_version"],
        region=run_kwargs["region"],
        save_root_flag_toggle=save_root_flag_toggle,
        save_systs_flag_toggle=save_systs_flag_toggle,
        bb_preselection_toggle=bb_preselection_toggle,
    )

    write_text(sh_path, dedent(content), mode=0o755)


def generate_for_one_combo(
    args, year: str, sample: str, subsample: str, files_per_job: int, nfiles_hint: int | None = None
):
    index_obj = load_index_json(year)
    nfiles = nfiles_hint or infer_nfiles(index_obj, sample, subsample)

    if nfiles is None and args.njobs is None and args.nfiles is None:
        # allow explicit overrides
        nfiles = args.nfiles

    if nfiles is None and args.njobs is None:
        raise RuntimeError(
            f"Cannot infer nfiles for {year}/{sample}/{subsample}. "
            f"Provide --nfiles or --njobs, or ensure data/index_{year}.json has counts."
        )

    if args.njobs is not None:
        njobs = args.njobs
    else:
        njobs = ceil_div(int(nfiles), files_per_job)

    print(
        f"[PLAN] year={year} sample={sample} subsample={subsample} nfiles={nfiles} files_per_job={files_per_job} -> njobs={njobs}"
    )

    # Layout root
    for job_idx in range(njobs):
        starti = job_idx * files_per_job
        endi = (
            starti + files_per_job if nfiles is None else min(starti + files_per_job, int(nfiles))
        )

        base_rel, logs_dir, exe_rel, sub_rel = make_paths(
            tag=args.tag,
            nano_version=args.nano_version,
            region=args.region,
            year=year,
            subsample=subsample,
            job_index=job_idx,
        )

        # JDL
        render_jdl(
            sub_path=sub_rel,
            exe_rel=exe_rel,
            logs_dir=logs_dir,
            x509_path=args.x509,
            request_memory=args.request_memory,
            singularity_image=args.singularity_image,
            job_flavour=args.job_flavour,
        )

        # SH
        run_kwargs = dict(
            starti=starti,
            endi=endi,
            batch_size=args.batch_size,
            sample=sample,
            subsample=subsample,
            processor=args.processor,
            maxchunks=args.maxchunks,
            chunksize=args.chunksize,
            save_root=args.save_root,
            save_systematics=args.save_systematics,
            fatjet_bb_preselection=args.fatjet_bb_preselection,
            nano_version=args.nano_version,
            region=args.region,
        )
        render_sh(
            sh_path=exe_rel,
            eos_user_path=args.eos_user_path,
            tag=args.tag,
            nano_version=args.nano_version,
            region=args.region,
            year=year,
            subsample=subsample,
            job_index=job_idx,
            git_user=args.git_user,
            git_branch=args.git_branch,
            run_kwargs=run_kwargs,
        )

    print(f"[OK] Generated {njobs} jobs for {year}/{sample}/{subsample}")


def generate_run_submit_sh(
    tag: str, nano_version: str, region: str, year: str, subsamples: list[str], eos_user_path: str
):
    submit_dir = Path("condor") / "skimmer" / f"{tag}_{nano_version}_{region}"
    run_sh_path = submit_dir / "run_submit.sh"

    lines = [
        "#!/bin/bash",
        "set -e",
        'echo "Auto-generated run_submit.sh"',
        "",
    ]

    lines.append("mkdir -p " + str(submit_dir) + "/logs")

    for subsample in subsamples:
        eos_base = f"/eos/user/{eos_user_path}/bbtautau/skimmer/{tag}_{nano_version}_{region}/{year}/{subsample}"
        for subdir in ["pickles", "parquet", "root", "jobchecks"]:
            lines.append(f"xrdfs root://eosuser.cern.ch/ mkdir -p {eos_base}/{subdir}")

    lines.append("for f in " + str(submit_dir) + "/*_eos.sub; do")
    lines.append('    condor_submit "$f"')
    lines.append("done")

    run_sh_path.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(run_sh_path, 0o755)
    print(f"[OK] Generated {run_sh_path}")


# -----------------------------
# CLI
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate batch .sub (JDL) + _eos.sh like submit.py, targeting CERN EOS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required identity
    parser.add_argument("--tag", required=True, help="e.g. 24Nov7Signal")
    parser.add_argument(
        "--year", nargs="+", required=True, help="One or more years, e.g. 2022 2018"
    )
    parser.add_argument("--samples", nargs="+", default=["HHbbtt"])
    parser.add_argument(
        "--subsamples",
        nargs="+",
        required=False,
        help="If omitted and YAML provided, read from YAML",
    )

    # Split control
    parser.add_argument(
        "--files-per-job", type=int, default=5, help="Default files per job (overridable via YAML)"
    )
    parser.add_argument("--njobs", type=int, default=None, help="Override number of jobs directly")
    parser.add_argument(
        "--nfiles",
        type=int,
        default=None,
        help="Override total number of files when index JSON is unavailable",
    )

    # run.py arguments
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=10000)
    parser.add_argument("--maxchunks", type=int, default=0)
    parser.add_argument("--processor", default="skimmer")
    parser.add_argument("--nano-version", default="v12_private")
    parser.add_argument("--region", default="signal")
    parser.add_argument(
        "--fatjet-bb-preselection", dest="fatjet_bb_preselection", action="store_true"
    )
    parser.add_argument(
        "--no-fatjet-bb-preselection", dest="fatjet_bb_preselection", action="store_false"
    )
    parser.set_defaults(fatjet_bb_preselection=False)
    parser.add_argument("--save-root", dest="save_root", action="store_true")
    parser.add_argument("--no-save-root", dest="save_root", action="store_false")
    parser.set_defaults(save_root=False)
    parser.add_argument("--save-systematics", dest="save_systematics", action="store_true")
    parser.add_argument("--no-save-systematics", dest="save_systematics", action="store_false")
    parser.set_defaults(save_systematics=False)

    # Infra / env
    parser.add_argument(
        "--x509", required=True, help="e.g. /afs/cern.ch/user/j/jinwa/x509up_u154433"
    )
    parser.add_argument(
        "--eos-user-path", default="c/cquarant", help="EOS path fragment like j/jinwa"
    )
    parser.add_argument("--git-user", default="jinwang137")
    parser.add_argument("--git-branch", default="main")

    # JDL knobs
    parser.add_argument("--request-memory", type=int, default=4500)
    parser.add_argument(
        "--singularity-image",
        default="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.8",
    )
    parser.add_argument("--job-flavour", default="workday")

    # YAML batch option (compatible with your prior flow)
    parser.add_argument("--yaml", help="YAML with overrides per year/sample/subsample")

    args = parser.parse_args()

    all_subsamples_global = []

    if args.yaml:
        ypath = Path(args.yaml)
        if not ypath.exists():
            sys.exit(f"[ERR] YAML not found: {ypath}")
        with ypath.open() as f:
            yobj = yaml.safe_load(f)

        tag_backup = args.tag
        for year in args.year:
            if year not in yobj:
                print(f"[WARN] Year {year} not found in YAML; using full YAML as fallback.")
                tdict = yobj
            else:
                tdict = yobj[year]

            print(f"[YAML] Submitting for year {year}")
            for sample, sdict in tdict.items():
                subsamples = sdict.get("subsamples", [])
                all_subsamples_global.extend(subsamples)
                files_per_job = sdict["files_per_job"]
                # optional knobs (fallback to CLI defaults if missing)
                args.maxchunks = sdict.get("maxchunks", args.maxchunks)
                args.chunksize = sdict.get("chunksize", args.chunksize)
                args.batch_size = sdict.get("batch_size", args.batch_size)
                args.tag = tag_backup  # keep outer tag

                if isinstance(files_per_job, dict):
                    for subsample in subsamples:
                        fpj = files_per_job[subsample]
                        generate_for_one_combo(
                            args=args,
                            year=str(year),
                            sample=sample,
                            subsample=subsample,
                            files_per_job=int(fpj),
                        )
                else:
                    for subsample in subsamples:
                        generate_for_one_combo(
                            args=args,
                            year=str(year),
                            sample=sample,
                            subsample=subsample,
                            files_per_job=int(files_per_job),
                        )
            generate_run_submit_sh(
                args.tag,
                args.nano_version,
                args.region,
                year,
                all_subsamples_global,
                args.eos_user_path,
            )
        return

    # Non-YAML path: use CLI samples/subsamples
    if not args.subsamples:
        sys.exit("[ERR] Must provide --subsamples (or use --yaml).")

    for year in args.year:
        print(f"[CLI] Submitting for year {year}")
        for sample in args.samples:
            for subsample in args.subsamples:
                all_subsamples_global.append(subsample)
                generate_for_one_combo(
                    args=args,
                    year=str(year),
                    sample=sample,
                    subsample=subsample,
                    files_per_job=args.files_per_job,
                )
        generate_run_submit_sh(
            args.tag,
            args.nano_version,
            args.region,
            year,
            all_subsamples_global,
            args.eos_user_path,
        )


if __name__ == "__main__":
    main()
