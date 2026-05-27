import ROOT
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--channel", required=True)
parser.add_argument("--cut", required=True)
parser.add_argument("--path", required=True)
parser.add_argument("--fine_tune", default="")
args = parser.parse_args()

MODEL = args.model
CHANNEL = args.channel
CUT = args.cut
PATH = args.path
FINE_TUNE = args.fine_tune

if FINE_TUNE:
    MODEL = f"{MODEL}_{FINE_TUNE}"
quantile_names = {
    # -1.0: "observed",
    0.025: "minus2sigma",
    0.16: "minus1sigma",
    0.5: "median",
    0.84: "plus1sigma",
    0.975: "plus2sigma",
}

path = PATH

file = ROOT.TFile(path + "/higgsCombineTest.AsymptoticLimits.mH120.root")
tree = file.Get("limit")

limits = {}
for i in range(tree.GetEntries()):
    tree.GetEntry(i)
    q = round(tree.quantileExpected, 3)
    name = quantile_names.get(q, str(q))
    if name == "median":
        limits = {"limit": tree.limit, "limitErr": tree.limitErr}

json_path = f"/work/kfuss/SVJ_GNN/output/{CHANNEL}_limits.json"
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        result = json.load(f)
else:
    result = {}

# Update nested structure
if CHANNEL not in result:
    result[CHANNEL] = {}
if MODEL not in result[CHANNEL]:
    result[CHANNEL][MODEL] = {}
result[CHANNEL][MODEL][CUT] = limits

for ch in result:
    for model in result[ch]:
        result[ch][model] = dict(sorted(result[ch][model].items(), key=lambda x: float(x[0])))

    def model_sort_key(model_name):
        is_benchmark = 0 if "benchmark" in model_name.lower() else 1
        best_limit = min(
            (v.get("limit", float("inf")) for v in result[ch][model_name].values()),
            default=float("inf")
        )
        return (is_benchmark, best_limit)

    result[ch] = dict(sorted(result[ch].items(), key=lambda x: model_sort_key(x[0])))

with open(json_path, "w") as f:
    json.dump(result, f, indent=3)

print(json.dumps({CHANNEL: {MODEL: {CUT: limits}}}, indent=3))