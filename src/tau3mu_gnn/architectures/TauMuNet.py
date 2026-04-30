import numpy as np
import random

import torch
import torch.nn as nn

from torch_geometric.nn.conv import GINEConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool

class TauMuNet(nn.Module):

    def __init__(
            self,
            conv_params,
            dropout,
            fc_params,
            num_node_features = 8, #8 # as we want to normally use also dzd0
            num_edge_features = 4, 
            num_graph_level_features=13,
            use_fusion = True,
            gnn_mode = "final_layer", # can be "BDT", "final_layer", "embed_once", "embed_always"
            scoring=False,
            num_classes = 1,
            scale_down_params=[],
            conv_type="GINE",   
            num_heads=4, 
            ):
        super(TauMuNet, self).__init__()

        self.conv_params = conv_params
        self.fc_params = fc_params
        self.use_fusion = use_fusion
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_graph_level_features = num_graph_level_features
        self.dropout = dropout
        self.embed_input = gnn_mode in ["embed_once", "embed_always"]
        self.embed_every_layer = gnn_mode == "embed_always"
        self.embed_final = gnn_mode == "final_layer"
        self.scoring = scoring
        self.scale_down = True if scale_down_params else False
        self.scale_down_params = scale_down_params
        self.conv_type = conv_type
        self.num_heads = num_heads

        self.input_dims = num_node_features

        if self.embed_input:
            self.input_dims += self.num_graph_level_features
        
        assert not self.scale_down or self.use_fusion, "scale_down requires use_fusion"

        #batch normalization features
        self.bn_fts = nn.BatchNorm1d(self.num_node_features)
        self.bn_e_fts = nn.BatchNorm1d(self.num_edge_features)
        self.bn_g_fts = nn.BatchNorm1d(self.num_graph_level_features)

        self.edge_convs = nn.ModuleList()

        # First layer: from input_dims to conv_params[0]
        self.edge_convs.append(self._build_conv(self.input_dims, conv_params[0]))

        for in_feat, out_feat in zip(conv_params[:-1], conv_params[1:]):
            if self.embed_every_layer:
                in_feat += self.num_graph_level_features
            self.edge_convs.append(self._build_conv(in_feat, out_feat))

        if self.use_fusion:
            in_chn = sum(x for x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Linear(in_chn, out_chn), nn.ReLU(), nn.BatchNorm1d(out_chn))
        
            scale_down_fcs = []
            if self.scale_down:
                for idx, (channels, drop_rate) in enumerate(scale_down_params):
                    if idx == 0:
                        in_chn = out_chn
                    else:
                        in_chn = scale_down_params[idx - 1][0]
                    
                    scale_down_fcs.append(nn.Sequential(
                        nn.Linear(in_chn, channels),
                        nn.ReLU(),
                        nn.BatchNorm1d(channels),
                        nn.Dropout(drop_rate)
                    ))
                out_chn = scale_down_params[-1][0]
                self.scale_down_fc = nn.Sequential(*scale_down_fcs)

        fcs = []
        for idx, (channels, drop_rate) in enumerate(fc_params):
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1]  
                if self.embed_final:
                    in_chn += num_graph_level_features
            else:
                in_chn = fc_params[idx - 1][0]
            
            fcs.append(nn.Sequential(
                nn.Linear(in_chn, channels),
                nn.ReLU(),
                nn.BatchNorm1d(channels),
                nn.Dropout(drop_rate)
            ))
        # Isnt this way and also in the old code the final layer applied twice? Once with ReLU and Dropout and once without. Is this intended?
        # Should the last layer have BatchNorm or do we want to take the raw embedding
        # Final fc layer 
        fcs.append(nn.Sequential(
            nn.Linear(fc_params[-1][0], num_classes),
            # nn.BatchNorm1d(num_classes)
        ))

        self.fc = nn.Sequential(*fcs)

    def _build_conv(self, in_feat, out_feat):
        if self.conv_type == "GINE":
            nns = nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.Dropout(self.dropout),
                nn.ReLU(),
                nn.Linear(out_feat, out_feat),
                nn.ReLU(),
                nn.BatchNorm1d(out_feat),
                nn.Dropout(self.dropout),
            )
            return GINEConv(nn=nns, edge_dim=self.num_edge_features)

        elif self.conv_type == "GAT":
            # concat=False → output dim stays out_feat (not out_feat * num_heads)
            return GATConv(
                in_channels=in_feat,
                out_channels=out_feat,
                heads=self.num_heads,
                edge_dim=self.num_edge_features,
                dropout=self.dropout,
                concat=False,
                residual=True,
            )

        elif self.conv_type == "Transformer":
            # concat=False → output dim stays out_feat (not out_feat * num_heads)
            return TransformerConv(
                in_channels=in_feat,
                out_channels=out_feat,
                heads=self.num_heads,
                edge_dim=self.num_edge_features,
                dropout=self.dropout,
                concat=False,
            )
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")
    
    def forward(self, data):
    
        x, edge_index, edge_attr, graph_feat = data.x, data.edge_index, data.edge_attr, data.graph_level_features
        #batch normalization features
        x = self.bn_fts(x)
        edge_attr = self.bn_e_fts(edge_attr)
        graph_feat = self.bn_g_fts(graph_feat)

        if self.embed_input:
            if data.batch is None:
                graph_feat = graph_feat.repeat(x.size(0), 1)
            else:
                graph_feat = graph_feat[data.batch]
            x = torch.cat([x, graph_feat], dim=-1)

        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            if idx > 0 and self.embed_every_layer:
                x = torch.cat([x, graph_feat], dim=-1)
            x = conv(x, edge_index, edge_attr)
            if self.use_fusion:
                outputs.append(x)
        #fuse outputs from all GINEConv layers
        if self.use_fusion:
            x = self.fusion_block(torch.cat(outputs, dim=1))

        #Global average pooling
        x = global_mean_pool(x, batch = data.batch) 
    
        if self.scale_down: 
            x = self.scale_down_fc(x)

        if self.embed_final:
            x = torch.cat([x, graph_feat], dim = -1)
        x = self.fc(x)
        if not self.scoring:
            x = torch.sigmoid(x)
        return x