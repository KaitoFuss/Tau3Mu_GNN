from tau3mu_gnn.graph.ConeGraphProducer import ConeGraphProducer
# from tau3mu_gnn.graph.ConeGraphProducer_Dataset import ConeGraphProducer_Dataset
import logging
import argparse

from tau3mu_gnn.graph.naming import graph_suffix
import os

parser = argparse.ArgumentParser()
parser.add_argument("--graph_type", choices=["fully", "knn", "r_threshold"], required=True, help="Type of edge connection")
parser.add_argument("--k", type=int, default=None, help="Number of nearest neighbors for knn")
parser.add_argument("--delta_r", type=float, default=None, help="Distance threshold for r_threshold")
parser.add_argument("--tau_channel", type=str, default="W", help="Tau production channel ('W' or 'HF')")
parser.add_argument("--unsculpting_mode", type=str, default=None, help="Unsculting mode. Please choose 'permutation', 'split_pt_uniformly', 'delete_muons', None")
parser.add_argument("--bdt_preselection_cut", type=float, default=None, help="BDT ISO cut: only events with bdt_ISO >= cut, one graph per event (candidate 0); not set: no cut")
# parser.add_argument("--nevents", type=int, default=100, help="Max number of events to process")
args = parser.parse_args()

# default values
# max_events_to_process = None
max_events_to_process = 300000
use_relative_angles = True
save_edge_attributes = True
save_event_number = False
save_fitted_tau_mass = True
save_fitted_tau_idx = False
verbose = True
use_displacement = True
cone_size = 0.5
unsculpting_mode = args.unsculpting_mode
# if use_muon_permutation:
#     max_events_to_process = max_events_to_process // 5

def create_graph(
    root: str,
    output_ident: str,
    label: float,
    mask: bool,
    edge_connection_mode: str,
    max_events_to_process: int,
    k_knn: int = None,
    delta_r_threshold: float = None,
    slimmed: bool = False,
    shuffle: bool = False,
    weights = None,
    tau_channel: str = "W",
    graph_level_branches: list = [],
    save_benchmark_score: bool = False,
    bdt_preselection_cut: float | None = None,
):
    suffix = graph_suffix(use_displacement, cone_size, edge_connection_mode, k=k_knn, delta_r=delta_r_threshold,
                          unsculpting_mode=unsculpting_mode)
    output_dataset_name = output_ident + suffix

    print(output_dataset_name)

    ConeGraphProducer(
        root,
        output_dataset_name,
        cone_size,
        edge_connection_mode=edge_connection_mode,
        delta_r_threshold=delta_r_threshold,
        k_knn=k_knn,
        max_events_to_process=max_events_to_process,
        use_relative_angles=use_relative_angles,
        save_edge_attributes=save_edge_attributes,
        save_event_number=save_event_number,
        save_fitted_tau_mass=save_fitted_tau_mass,
        save_benchmark_score=save_benchmark_score,
        weights=weights,
        save_fitted_tau_idx=save_fitted_tau_idx,
        mask=mask,
        label=label,
        verbose=verbose,
        tau_channel=tau_channel,
        graph_level_branches=graph_level_branches,
        slimmed=slimmed,
        shuffle_bool=shuffle,
        use_displacement=use_displacement,
        unsculpting_mode=unsculpting_mode,
        bdt_preselection_cut=bdt_preselection_cut,
    )
                


if __name__ == "__main__":
    model_type = args.graph_type
    k = args.k
    delta_r = args.delta_r
    tau_channel = args.tau_channel
    bdt_preselection_cut = args.bdt_preselection_cut
    # max_events_to_process = args.nevents

    if model_type == "knn" and k is None:
        raise ValueError("For 'knn' model_type, --k must be specified.")
    if model_type == "r_threshold" and delta_r is None:
        raise ValueError("For 'r_threshold' model_type, --delta_r must be specified.")
    if tau_channel not in ["W", "HF"]:
        raise ValueError("Unsupported tau_channel. Please choose either 'W' or 'HF'.")

    # root_W_signal = "data/signal_W" 
    # output_ident_signal = "W_decay_signal"
    # create_graph(
    #     root_W_signal, 
    #     output_ident_signal, 
    #     label=1., 
    #     mask=False, 
    #     edge_connection_mode=model_type,
    #     k_knn=k, 
    #     delta_r_threshold=delta_r
    # )

    # root_bkg = "data/background" 
    # output_ident_bkg = "background"
    # create_graph(
    #     root_bkg, 
    #     output_ident_bkg, 
    #     label=0., 
    #     mask=True, 
    #     edge_connection_mode=model_type,
    #     k_knn=k, 
    #     delta_r_threshold=delta_r
    # )

    prefix = "Selected_"
    graph_file_ident = "skimmed"
    if bdt_preselection_cut is not None:
        graph_file_ident += f"_BDT_cut_{bdt_preselection_cut}"

    if tau_channel == "W":
        base_direc_W_signal = "src/tau3mu_gnn/data/signal_W"
        if bdt_preselection_cut is not None:
            base_direc_W_signal += "/bdt_full"
        # base_direc_W_signal = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W"
        root_W_signals_skimmed = [
            [base_direc_W_signal, graph_file_ident],
        ]
        W_graph_level_branches = [
            f'{prefix}TauTo3Mu_fitted_pt',
            'TauPlusMET_Tau_Puppi_mT',
            #'cand_refit_tau_dBetaIsoCone0p8strength0p2_rel',
            #'deltaPhi(TauPlusMET_Puppi_phi[0], Slimmed_TauTo3Mu_fitted_phi[0])',
            'TauPlusMET_PuppiMET_pt',
            #'Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)', # only for >= v4
        #     'cand_refit_tau_pt/cand_refit_met_pt', # only for >= v2
        #     'cand_refit_dRtauMuonMax',
            'TauPlusMET_Puppi_pt',
            'TauPlusMET_PuppiMETminPz',
            'TauPlusMET_PuppiMETmaxPz',
            f'{prefix}TauTo3Mu_dZmu12', 
            f'{prefix}TauTo3Mu_dZmu13', 
            f'{prefix}TauTo3Mu_dZmu23',
            f'{prefix}TauTo3Mu_sigLxy_3muVtxBS',
            f'{prefix}TauTo3Mu_vtx_chi2*({prefix}TauTo3Mu_vtx_Ndof**-1)',
            f'{prefix}TauTo3Mu_CosAlpha2D_LxyP3mu',
        ]

        for base_direc, signal_identifier in root_W_signals_skimmed:
            output_ident_signal = "W_decay_signal_" + signal_identifier.replace("/", "")
            root_W_signal = base_direc
            create_graph(
                root_W_signal, 
                output_ident_signal, 
                label=1., 
                mask=False, 
                edge_connection_mode=model_type,
                max_events_to_process=max_events_to_process,
                k_knn=k, 
                delta_r_threshold=delta_r,
                slimmed = True,
                shuffle = True,
                tau_channel=tau_channel,
                graph_level_branches=W_graph_level_branches,
                weights="npvs_rescale",
                save_benchmark_score=True,
                bdt_preselection_cut=bdt_preselection_cut
            )
        
        base_direc_W_bkg = "src/tau3mu_gnn/data/background_W"
        if bdt_preselection_cut is not None:
            base_direc_W_bkg += "/bdt_full"
        # base_direc_W_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W"
    
        root_bkgs_skimmed = [
            [base_direc_W_bkg, graph_file_ident]
        ]
        for base_direc, bkg_identifier in root_bkgs_skimmed:
            output_ident_bkg = "W_decay_background_" + bkg_identifier.replace("/", "")
            root_bkg = base_direc
            create_graph(
                root_bkg, 
                output_ident_bkg, 
                label=0., 
                mask=True, 
                edge_connection_mode=model_type,
                max_events_to_process=max_events_to_process,
                k_knn=k, 
                delta_r_threshold=delta_r,
                slimmed = True,
                shuffle = True,
                tau_channel=tau_channel,
                graph_level_branches=W_graph_level_branches,
                save_benchmark_score=True,
                bdt_preselection_cut=bdt_preselection_cut
            )
    
    elif tau_channel == "HF":
        # base_direc_HF_signal = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF"
        base_direc_HF_signal = "src/tau3mu_gnn/data/signal_HF"
        if bdt_preselection_cut is not None:
            base_direc_HF_signal += "/bdt_full"

        root_HF_signals_skimmed = [
            [base_direc_HF_signal, graph_file_ident],
        ]
        HF_graph_level_branches = [
            f'Muon_pt[{prefix}TauTo3Mu_mu3_idx]',
            
            # 'Muon_mediumId[Slimmed_TauTo3Mu_mu1_idx]', 
            # 'Muon_mediumId[Slimmed_TauTo3Mu_mu2_idx]',
            # 'Muon_mediumId[Slimmed_TauTo3Mu_mu3_idx]',

            f'Muon_isGlobal[{prefix}TauTo3Mu_mu1_idx]', 
            f'Muon_isGlobal[{prefix}TauTo3Mu_mu2_idx]', 
            f'Muon_isGlobal[{prefix}TauTo3Mu_mu3_idx]', 

            f'{prefix}TauTo3Mu_dZmu12', 
            f'{prefix}TauTo3Mu_dZmu13', 
            f'{prefix}TauTo3Mu_dZmu23', 

            f'{prefix}TauTo3Mu_sigLxy_3muVtxBS', 
            f'{prefix}TauTo3Mu_CosAlpha2D_LxyP3mu', 

            f'{prefix}TauTo3Mu_vtx_chi2*({prefix}TauTo3Mu_vtx_Ndof**-1)',
        ]

        for base_direc, signal_identifier in root_HF_signals_skimmed:
            output_ident_signal = "HF_decay_signal_" + signal_identifier.replace("/", "")
            root_HF_signal = base_direc
            create_graph(
                root_HF_signal, 
                output_ident_signal, 
                label=1., 
                mask=False, 
                edge_connection_mode=model_type,
                max_events_to_process=max_events_to_process,
                k_knn=k, 
                delta_r_threshold=delta_r,
                slimmed=True,
                shuffle=True,
                tau_channel=tau_channel,
                graph_level_branches=HF_graph_level_branches,
                weights="npvs_rescale",
                save_benchmark_score=False,
                bdt_preselection_cut=bdt_preselection_cut
            )

        # base_direc_HF_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF"
        base_direc_HF_bkg = "src/tau3mu_gnn/data/background_HF"

        if bdt_preselection_cut is not None:
            base_direc_HF_bkg += "/bdt_full"

        root_bkgs_skimmed = [
            [base_direc_HF_bkg, graph_file_ident]
        ]

        for base_direc, bkg_identifier in root_bkgs_skimmed:
            output_ident_bkg = "HF_decay_background_" + bkg_identifier.replace("/", "")
            root_bkg = base_direc
            create_graph(
                root_bkg, 
                output_ident_bkg, 
                label=0., 
                mask=True, 
                edge_connection_mode=model_type,
                max_events_to_process=max_events_to_process//4 if max_events_to_process is not None else max_events_to_process,
                k_knn=k, 
                delta_r_threshold=delta_r,
                slimmed=True,
                shuffle=True,
                tau_channel=tau_channel,
                graph_level_branches=HF_graph_level_branches,
                save_benchmark_score=False,
                bdt_preselection_cut=bdt_preselection_cut
            )

    # root_W_signals = [
    #    [base_direc_W_signal, "winter24"],
    #    [base_direc_W_signal, "summer24"],
    # ]
    # for base_direc, signal_identifier in root_W_signals:
    #     output_ident_signal = "W_decay_signal_" + signal_identifier.replace("/", "")
    #     root_W_signal = base_direc + signal_identifier
    #     create_graph(
    #     root_W_signal, 
    #     output_ident_signal, 
    #     label=1., 
    #     mask=False, 
    #     edge_connection_mode=model_type,
    #     k_knn=k, 
    #     delta_r_threshold=delta_r,
    #     slimmed=False,
    # )


    # root_bkgs = [
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024D-PromptReco-v1_MINIAOD/", "250430_100827/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024F-PromptReco-v1_MINIAOD/", "250430_100852/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024I-PromptReco-v1_MINIAOD/", "250430_100917/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024B-PromptReco-v1_MINIAOD/", "250430_100811/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024E-PromptReco-v1_MINIAOD/", "250430_100835/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024G-PromptReco-v1_MINIAOD/", "250430_100900/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024I-PromptReco-v2_MINIAOD/", "250430_100925/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024C-PromptReco-v1_MINIAOD/", "250430_100819/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024E-PromptReco-v2_MINIAOD/", "250430_100843/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024H-PromptReco-v1_MINIAOD/", "250430_100908/0000"],

    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024D-PromptReco-v1_MINIAOD/", "250430_100949/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024F-PromptReco-v1_MINIAOD/", "250430_101014/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024I-PromptReco-v1_MINIAOD/", "250430_101038/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024B-PromptReco-v1_MINIAOD/", "250430_100933/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024E-PromptReco-v1_MINIAOD/", "250430_100957/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024G-PromptReco-v1_MINIAOD/", "250430_101022/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024I-PromptReco-v2_MINIAOD/", "250430_101046/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024C-PromptReco-v1_MINIAOD/", "250430_100941/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024E-PromptReco-v2_MINIAOD/", "250430_101005/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024H-PromptReco-v1_MINIAOD/", "250430_101030/0000"],
        
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024D-PromptReco-v1_MINIAOD/", "250430_101111/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024F-PromptReco-v1_MINIAOD/", "250430_101135/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024I-PromptReco-v1_MINIAOD/", "250430_101159/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024B-PromptReco-v1_MINIAOD/", "250430_101054/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024E-PromptReco-v1_MINIAOD/", "250430_101119/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024G-PromptReco-v1_MINIAOD/", "250430_101143/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024I-PromptReco-v2_MINIAOD/", "250430_101207/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024C-PromptReco-v1_MINIAOD/", "250430_101102/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024E-PromptReco-v2_MINIAOD/", "250430_101127/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024H-PromptReco-v1_MINIAOD/", "250430_101151/0000"],
        
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024D-PromptReco-v1_MINIAOD/", "250430_101232/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024F-PromptReco-v1_MINIAOD/", "250430_101256/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024I-PromptReco-v1_MINIAOD/", "250430_101320/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024B-PromptReco-v1_MINIAOD/", "250430_101215/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024E-PromptReco-v1_MINIAOD/", "250430_101240/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024G-PromptReco-v1_MINIAOD/", "250430_101304/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024I-PromptReco-v2_MINIAOD/", "250430_101328/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024C-PromptReco-v1_MINIAOD/", "250430_101223/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024E-PromptReco-v2_MINIAOD/", "250430_101248/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024H-PromptReco-v1_MINIAOD/", "250430_101312/0000"],

    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024D-PromptReco-v1_MINIAOD/", "250430_101352/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024F-PromptReco-v1_MINIAOD/", "250430_101416/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024I-PromptReco-v1_MINIAOD/", "250430_101443/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024B-PromptReco-v1_MINIAOD/", "250430_101336/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024E-PromptReco-v1_MINIAOD/", "250430_101400/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024G-PromptReco-v1_MINIAOD/", "250430_101424/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024I-PromptReco-v2_MINIAOD/", "250430_101451/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024C-PromptReco-v1_MINIAOD/", "250430_101344/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024E-PromptReco-v2_MINIAOD/", "250430_101408/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024H-PromptReco-v1_MINIAOD/", "250430_101433/0000"],
        
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024B-PromptReco-v1_MINIAOD/", "250430_101459/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024D-PromptReco-v1_MINIAOD/", "250430_101515/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024E-PromptReco-v2_MINIAOD/", "250430_101531/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024G-PromptReco-v1_MINIAOD/", "250430_101547/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024I-PromptReco-v1_MINIAOD/", "250430_101603/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024C-PromptReco-v1_MINIAOD/", "250430_101507/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024E-PromptReco-v1_MINIAOD/", "250430_101523/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024F-PromptReco-v1_MINIAOD/", "250430_101539/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024H-PromptReco-v1_MINIAOD/", "250430_101555/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024I-PromptReco-v2_MINIAOD/", "250430_101611/0000"],
        

    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024B-PromptReco-v1_MINIAOD/", "250430_101619/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024D-PromptReco-v1_MINIAOD/", "250430_101635/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024E-PromptReco-v2_MINIAOD/", "250430_101652/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024G-PromptReco-v1_MINIAOD/", "250430_101711/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024I-PromptReco-v1_MINIAOD/", "250430_101727/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024C-PromptReco-v1_MINIAOD/", "250430_101627/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024E-PromptReco-v1_MINIAOD/", "250430_101643/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024F-PromptReco-v1_MINIAOD/", "250430_101700/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024H-PromptReco-v1_MINIAOD/", "250430_101719/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024I-PromptReco-v2_MINIAOD/", "250430_101735/0000"],

    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024B-PromptReco-v1_MINIAOD/", "250430_101743/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024D-PromptReco-v1_MINIAOD/", "250430_101759/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024E-PromptReco-v2_MINIAOD/", "250430_101815/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024G-PromptReco-v1_MINIAOD/", "250430_101831/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024I-PromptReco-v1_MINIAOD/", "250430_101847/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024C-PromptReco-v1_MINIAOD/", "250430_101751/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024E-PromptReco-v1_MINIAOD/", "250430_101807/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024F-PromptReco-v1_MINIAOD/", "250430_101823/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024H-PromptReco-v1_MINIAOD/", "250430_101839/0000"],
    #     [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024I-PromptReco-v2_MINIAOD/", "250430_101855/0000"],
    # ]
    # for base_direc, bkg_identifier, tail in root_bkgs:
    #     output_ident_bkg = "background_" + bkg_identifier.replace("/", "")
    #     root_bkg = base_direc + bkg_identifier + tail
    #     create_graph(
    #     root_bkg, 
    #     output_ident_bkg, 
    #     label=0., 
    #     mask=True, 
    #     edge_connection_mode=model_type,
    #     k_knn=k, 
    #     delta_r_threshold=delta_r
    # )