#!/bin/bash
# Step 14: copy the T3 files of a file list (from t3_list_files.py) to EOS. Run on lxplus.
#
#   voms-proxy-init --voms cms --valid 192:00      # T3 read needs a grid proxy
#   tmux                                           # long copy: keep it alive (note the lxplus node name)
#   jobs/migration/copy_to_eos.sh filelist.tsv [group ...]
#
# A file is skipped if its EOS copy has the same size, so a second run continues a broken copy.
# PARALLEL copies at the same time (default 4). Failed files go to copy_failed.tsv; run again with that list.
# /eos/cms/... targets go through eoscms, /eos/user/... through eosuser. In the group space (/eos/cms/store/group/
# phys_bphys) the e-group may not delete: a broken copy there may need a cleaner (cms-eos-pag-bphysics-cleaners).
set -euo pipefail

LIST=${1:?usage: copy_to_eos.sh filelist.tsv [group ...]}
shift
GROUPS_WANTED=" $* "
T3=root://t3se01.psi.ch:1094   # public T3 door; t3dcachedb03 is reachable only inside PSI
# The lxplus xrdcp (5.9.8) fails the GSI login of the T3 dCache ("malformed random tag ... refuse to sign");
# 5.5-5.8 work (tested 2026-10-30). XRD: folder of the xrootd build to use.
XRD=${XRD:-/cvmfs/sft.cern.ch/lcg/releases/xrootd/5.8.3-1e12b/x86_64-el9-gcc11-opt}
export PATH="$XRD/bin:$PATH" LD_LIBRARY_PATH="$XRD/lib64:$XRD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PARALLEL=${PARALLEL:-4}
FAILED=copy_failed.tsv

voms-proxy-info --exists --valid 1:00 >/dev/null 2>&1 || { echo "no valid grid proxy: voms-proxy-init --voms cms --valid 192:00"; exit 1; }
: > "$FAILED"

copy_one() {   # group bytes t3_path eos_path
    local group=$1 bytes=$2 src=$3 dest=$4
    if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$bytes" ]; then
        return 0
    fi
    for attempt in 1 2 3; do
        case $dest in
            /eos/cms/*) mgm=root://eoscms.cern.ch ;;
            *)          mgm=root://eosuser.cern.ch ;;
        esac
        if xrdcp --force --path --nopbar --cksum adler32 "$T3/$src" "$mgm/$dest" 2>>copy_errors.log; then
            echo "ok   $dest"
            return 0
        fi
        sleep $((60 * attempt))
    done
    echo "FAIL $dest"
    printf '%s\t%s\t%s\t%s\n' "$group" "$bytes" "$src" "$dest" >> "$FAILED"
}
export -f copy_one
export T3 FAILED

awk -F'\t' -v want="$GROUPS_WANTED" 'want == "  " || index(want, " " $1 " ")' "$LIST" |
    xargs -d '\n' -P "$PARALLEL" -n 1 bash -c 'IFS=$'"'"'\t'"'"' read -r g b s d <<< "$1"; copy_one "$g" "$b" "$s" "$d"' _

n=$(wc -l < "$FAILED")
echo "done; $n failed (list: $FAILED)"
[ "$n" -eq 0 ]
