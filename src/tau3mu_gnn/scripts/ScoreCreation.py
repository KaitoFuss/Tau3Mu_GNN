"""Old entry point: score every .root file of --root with each model; outputs as before."""
import argparse
import os

import torch

from tau3mu_gnn.gnn_training.model_loading import load_model
from tau3mu_gnn.gnn_training.ScoreProducer import score_events


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='Folder to read the data from')
    parser.add_argument('--output_folder', type=str, required=True, help="Output to store the files with the scores")
    parser.add_argument('--model_names', type=str, required=True, help="Name of GNN model", nargs='+')
    parser.add_argument('--archs', type=str, required=True, help="Architecture of GNN model", nargs='+')
    parser.add_argument('--tau_channel', type=str, required=True, help="Tau production channel")
    parser.add_argument('--gnn_mode', type=str, default="BDT", help="GNN Mode")
    parser.add_argument('--model', type=str, default="skimmed_dzd0_cone_size_0.5_knn_10", help="Which model to train/ which graph construction was used")
    parser.add_argument('--storage_folder', type=str, required=True, help="Output to store the root files with the scores")
    parser.add_argument('--max_events_to_process', type=int, default=None, help="max_events_to_process")
    parser.add_argument('--conv_types', type=str, required=True, help="Conv layer types", nargs='+')
    parser.add_argument("--unsculpting_mode", type=str, default=None, help="Unsculting mode. Please choose 'permutation', 'split_pt_uniformly', 'delete_muons', None")
    parser.add_argument("--bdt_preselection_cut", type=float, default=None, help="BDT preselection score cut")

    args = parser.parse_args(argv)

    assert len(args.model_names) == len(args.archs) == len(args.conv_types), (
        f"Number of model names ({len(args.model_names)}) must match number of architectures "
        f"({len(args.archs)}). Got model_names={args.model_names}, archs={args.archs}")
    if "signal" in args.root:
        mode = "signal"
    elif "background" in args.root:
        mode = "background"
    else:
        raise ValueError(f"--root must contain 'signal' or 'background': {args.root}")
    suffix = "" if args.bdt_preselection_cut is None else "_with_bdt_cut"
    in_files = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith(".root")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BDT Selections cut: {args.bdt_preselection_cut}")

    models, outputs, npz_paths = {}, {}, {}
    for model_name, conv_type in zip(args.model_names, args.conv_types):
        models[model_name] = load_model(args.output_folder, model_name, conv_type, args.gnn_mode,
                                        args.model, args.tau_channel, device)
        storage = os.path.join(args.storage_folder, model_name)
        outputs[model_name] = {f: os.path.join(storage, os.path.splitext(os.path.basename(f))[0] + "_with_scores.root")
                               for f in in_files}
        npz_paths[model_name] = os.path.join(args.output_folder, model_name,
                                             f"raw_scores_and_tau_mass_{mode}{suffix}.npz")
    score_events(in_files, models, outputs, npz_paths, args.tau_channel, args.unsculpting_mode,
                 args.bdt_preselection_cut, device=device, max_events_to_process=args.max_events_to_process)


if __name__ == "__main__":
    main()
