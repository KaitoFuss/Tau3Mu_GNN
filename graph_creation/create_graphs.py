from utils.ConeGraphProducer import ConeGraphProducer
import logging

edge_connection_modes = ["fully", "knn", "r_threshold"]
cone_sizes = [0.7]
k_knns = [3]
delta_r_thresholds = [0.2]
max_events_to_process = 1 # change to None later
save_edge_attributes = False # do we want edge features?
save_event_number = False
weights = None
save_fitted_tau_idx = False
verbose = True

def create_graph(root: str, label: float, mask: bool):
    for cone_size in cone_sizes:
        for edge_connection_mode in edge_connection_modes:
            if edge_connection_mode == "fully":
                delta_r_threshold = None
                k_knn = None
                output_dataset_name = "W_decay_signal_cone_size_" + str(cone_size) + edge_connection_mode
                ConeGraphProducer(
                        root, 
                        output_dataset_name, 
                        cone_size, 
                        edge_connection_mode=edge_connection_mode,
                        delta_r_threshold=delta_r_threshold,
                        k_knn=k_knn,
                        max_events_to_process=max_events_to_process,
                        save_edge_attributes=save_edge_attributes,
                        save_event_number=save_event_number,
                        weights=weights, 
                        save_fitted_tau_idx=save_fitted_tau_idx,
                        mask=mask,
                        label=label,
                        verbose=verbose
                    )
            elif edge_connection_mode == "knn":
                delta_r_threshold = None
                for k_knn in k_knns:
                    output_dataset_name = "W_decay_signal_cone_size_" + str(cone_size) + edge_connection_mode + "_" + str(k_knn)
                    ConeGraphProducer(
                        root, 
                        output_dataset_name, 
                        cone_size, 
                        edge_connection_mode=edge_connection_mode,
                        delta_r_threshold=delta_r_threshold,
                        k_knn=k_knn,
                        max_events_to_process=max_events_to_process,
                        save_edge_attributes=save_edge_attributes,
                        save_event_number=save_event_number,
                        weights=weights, 
                        save_fitted_tau_idx=save_fitted_tau_idx,
                        mask=mask,
                        label=label,
                        verbose=verbose
                    )
            elif edge_connection_mode == "r_threshold":
                k_knn = None
                for delta_r_threshold in delta_r_thresholds:
                    output_dataset_name = "W_decay_signal_cone_size_" + str(cone_size) + edge_connection_mode + "_" + str(delta_r_threshold)
                    ConeGraphProducer(
                        root, 
                        output_dataset_name, 
                        cone_size, 
                        edge_connection_mode=edge_connection_mode,
                        delta_r_threshold=delta_r_threshold,
                        k_knn=k_knn,
                        max_events_to_process=max_events_to_process,
                        save_edge_attributes=save_edge_attributes,
                        save_event_number=save_event_number,
                        weights=weights, 
                        save_fitted_tau_idx=save_fitted_tau_idx,
                        mask=mask,
                        label=label,
                        verbose=verbose
                    )
                


if __name__ == "__main__":
    root_W_signal = "/pnfs/psi.ch/cms/trivcat/store/user/manzoni/tau3mu_gnn/WTo3MuNu-smfull_TuneCP5_13p6TeV_madgraph-pythia8/_WTo3MuNu-smfull_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24MiniAOD-140X_mcRun3_2024_realistic_v26-v2_MINIAODSIM/250430_104612/0000/"
    create_graph(root_W_signal, label=1., mask=False)
    root_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/manzoni/tau3mu_gnn/ParkingDoubleMuonLowMass*/_ParkingDoubleMuonLowMass*_Run2024*_MINIAOD/*/*/*root"
    create_graph(root_bkg, label=0., mask=True)