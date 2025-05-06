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
                        weights=weights, 
                        save_fitted_tau_idx=save_fitted_tau_idx,
                        mask=mask,
                        label=label,
                        verbose=verbose,
                        use_displacement=use_displacement,
                    )
                


if __name__ == "__main__":
    root_W_signal = "data/signal_W" # change later to /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/signal_W
    output_ident = "W_decay_signal"
    create_graph(root_W_signal, output_ident, label=1., mask=False)
    root_bkg = "data/background" # change later to /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/background
    output_ident = "background"
    create_graph(root_bkg, output_ident, label=0., mask=True)