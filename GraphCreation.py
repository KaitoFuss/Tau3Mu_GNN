from utils.ConeGraphProducer import ConeGraphProducer
import logging

# edge_connection_modes = ["fully", "knn", "r_threshold"]
edge_connection_modes = ["fully"]
# test with delta_r
cone_sizes = [0.5]
k_knns = [3]
delta_r_thresholds = [0.2]
max_events_to_process = None
use_relative_angles = True
save_edge_attributes = True
save_event_number = False
save_fitted_tau_mass = True
weights = None # will need to be changed later for signal to adapt to data pileup distribution
save_fitted_tau_idx = False
verbose = True
use_displacement = True
dzd0 = "_dzd0" if use_displacement else ""

def create_graph(root: str, output_ident: str, label: float, mask: bool):
    for cone_size in cone_sizes:
        for edge_connection_mode in edge_connection_modes:
            if edge_connection_mode == "fully":
                delta_r_threshold = None
                k_knn = None
                output_dataset_name = output_ident + dzd0 +  "_cone_size_" + str(cone_size) + "_" + edge_connection_mode
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
                        weights=weights, 
                        save_fitted_tau_idx=save_fitted_tau_idx,
                        mask=mask,
                        label=label,
                        verbose=verbose,
                        use_displacement=use_displacement,
                    )
            elif edge_connection_mode == "knn":
                delta_r_threshold = None
                for k_knn in k_knns:
                    output_dataset_name = output_ident + dzd0 + "_cone_size_" + str(cone_size) + "_" + edge_connection_mode + "_" + str(k_knn)
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
                        weights=weights, 
                        save_fitted_tau_idx=save_fitted_tau_idx,
                        mask=mask,
                        label=label,
                        verbose=verbose,
                        use_displacement=use_displacement,
                    )
            elif edge_connection_mode == "r_threshold":
                k_knn = None
                for delta_r_threshold in delta_r_thresholds:
                    output_dataset_name = output_ident + dzd0 + "_cone_size_" + str(cone_size) + "_" + edge_connection_mode + "_" + str(delta_r_threshold)
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
                        weights=weights, 
                        save_fitted_tau_idx=save_fitted_tau_idx,
                        mask=mask,
                        label=label,
                        verbose=verbose,
                        use_displacement=use_displacement,
                    )
                


if __name__ == "__main__":
    root_W_signal = "data/signal_W" 
    output_ident = "W_decay_signal"
    create_graph(root_W_signal, output_ident, label=1., mask=False)

    root_bkg = "data/background" 
    output_ident = "background"
    create_graph(root_bkg, output_ident, label=0., mask=True)

    
    base_direc_W_signal = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/signal_W/"
    root_W_signals = [
        [base_direc_W_signal, "winter24"]
        [base_direc_W_signal, "summer24"]
    ]
    for base_direc, signal_identifier in root_W_signals:
        output_ident = "W_decay_signal_" + signal_identifier.replace("/", "")
        root_W_signal = base_direc + signal_identifier
        create_graph(root_W_signal, output_ident, label=1., mask=False)


    base_direc_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/background/"
    root_bkgs = [
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024D-PromptReco-v1_MINIAOD/", "250430_100827/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024F-PromptReco-v1_MINIAOD/", "250430_100852/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024I-PromptReco-v1_MINIAOD/", "250430_100917/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024B-PromptReco-v1_MINIAOD/", "250430_100811/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024E-PromptReco-v1_MINIAOD/", "250430_100835/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024G-PromptReco-v1_MINIAOD/", "250430_100900/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024I-PromptReco-v2_MINIAOD/", "250430_100925/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024C-PromptReco-v1_MINIAOD/", "250430_100819/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024E-PromptReco-v2_MINIAOD/", "250430_100843/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024H-PromptReco-v1_MINIAOD/", "250430_100908/0000"],

        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024D-PromptReco-v1_MINIAOD/", "250430_100949/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024F-PromptReco-v1_MINIAOD/", "250430_101014/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024I-PromptReco-v1_MINIAOD/", "250430_101038/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024B-PromptReco-v1_MINIAOD/", "250430_100933/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024E-PromptReco-v1_MINIAOD/", "250430_100957/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024G-PromptReco-v1_MINIAOD/", "250430_101022/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024I-PromptReco-v2_MINIAOD/", "250430_101046/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024C-PromptReco-v1_MINIAOD/", "250430_100941/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024E-PromptReco-v2_MINIAOD/", "250430_101005/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024H-PromptReco-v1_MINIAOD/", "250430_101030/0000"],
        
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024D-PromptReco-v1_MINIAOD/", "250430_101111/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024F-PromptReco-v1_MINIAOD/", "250430_101135/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024I-PromptReco-v1_MINIAOD/", "250430_101159/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024B-PromptReco-v1_MINIAOD/", "250430_101054/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024E-PromptReco-v1_MINIAOD/", "250430_101119/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024G-PromptReco-v1_MINIAOD/", "250430_101143/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024I-PromptReco-v2_MINIAOD/", "250430_101207/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024C-PromptReco-v1_MINIAOD/", "250430_101102/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024E-PromptReco-v2_MINIAOD/", "250430_101127/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024H-PromptReco-v1_MINIAOD/", "250430_101151/0000"],
        
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024D-PromptReco-v1_MINIAOD/", "250430_101232/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024F-PromptReco-v1_MINIAOD/", "250430_101256/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024I-PromptReco-v1_MINIAOD/", "250430_101320/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024B-PromptReco-v1_MINIAOD/", "250430_101215/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024E-PromptReco-v1_MINIAOD/", "250430_101240/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024G-PromptReco-v1_MINIAOD/", "250430_101304/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024I-PromptReco-v2_MINIAOD/", "250430_101328/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024C-PromptReco-v1_MINIAOD/", "250430_101223/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024E-PromptReco-v2_MINIAOD/", "250430_101248/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024H-PromptReco-v1_MINIAOD/", "250430_101312/0000"],

        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024D-PromptReco-v1_MINIAOD/", "250430_101352/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024F-PromptReco-v1_MINIAOD/", "250430_101416/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024I-PromptReco-v1_MINIAOD/", "250430_101443/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024B-PromptReco-v1_MINIAOD/", "250430_101336/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024E-PromptReco-v1_MINIAOD/", "250430_101400/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024G-PromptReco-v1_MINIAOD/", "250430_101424/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024I-PromptReco-v2_MINIAOD/", "250430_101451/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024C-PromptReco-v1_MINIAOD/", "250430_101344/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024E-PromptReco-v2_MINIAOD/", "250430_101408/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024H-PromptReco-v1_MINIAOD/", "250430_101433/0000"],
        
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024B-PromptReco-v1_MINIAOD/", "250430_101459/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024D-PromptReco-v1_MINIAOD/", "250430_101515/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024E-PromptReco-v2_MINIAOD/", "250430_101531/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024G-PromptReco-v1_MINIAOD/", "250430_101547/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024I-PromptReco-v1_MINIAOD/", "250430_101603/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024C-PromptReco-v1_MINIAOD/", "250430_101507/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024E-PromptReco-v1_MINIAOD/", "250430_101523/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024F-PromptReco-v1_MINIAOD/", "250430_101539/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024H-PromptReco-v1_MINIAOD/", "250430_101555/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024I-PromptReco-v2_MINIAOD/", "250430_101611/0000"],
        

        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024B-PromptReco-v1_MINIAOD/", "250430_101619/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024D-PromptReco-v1_MINIAOD/", "250430_101635/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024E-PromptReco-v2_MINIAOD/", "250430_101652/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024G-PromptReco-v1_MINIAOD/", "250430_101711/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024I-PromptReco-v1_MINIAOD/", "250430_101727/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024C-PromptReco-v1_MINIAOD/", "250430_101627/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024E-PromptReco-v1_MINIAOD/", "250430_101643/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024F-PromptReco-v1_MINIAOD/", "250430_101700/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024H-PromptReco-v1_MINIAOD/", "250430_101719/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024I-PromptReco-v2_MINIAOD/", "250430_101735/0000"],

        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024B-PromptReco-v1_MINIAOD/", "250430_101743/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024D-PromptReco-v1_MINIAOD/", "250430_101759/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024E-PromptReco-v2_MINIAOD/", "250430_101815/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024G-PromptReco-v1_MINIAOD/", "250430_101831/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024I-PromptReco-v1_MINIAOD/", "250430_101847/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024C-PromptReco-v1_MINIAOD/", "250430_101751/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024E-PromptReco-v1_MINIAOD/", "250430_101807/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024F-PromptReco-v1_MINIAOD/", "250430_101823/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024H-PromptReco-v1_MINIAOD/", "250430_101839/0000"],
        [base_direc_bkg, "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024I-PromptReco-v2_MINIAOD/", "250430_101855/0000"],
    ]
    for base_direc, bkg_identifier, tail in root_bkgs:
        output_ident = "background_" + bkg_identifier.replace("/", "")
        root_bkg = base_direc + bkg_identifier + tail
        create_graph(root_bkg, output_ident, label=0., mask=True)