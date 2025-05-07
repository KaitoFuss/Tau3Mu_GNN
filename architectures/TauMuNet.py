import numpy as np

import torch
import torch.nn as nn

from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn import global_mean_pool

class TauMuNet(nn.Module):

    def __init__(
            self,
            conv_params,
            fc_params,
            input_dims = 7, # as we want to normally use also dzd0
            num_edge_features = 4, 
            use_fusion = True,
            num_classes = 1,
            ):
        super(TauMuNet, self).__init__()

        self.conv_params = conv_params
        self.fc_params = fc_params
        self.use_fusion = use_fusion
        self.num_classes = num_classes
        self.input_dims = input_dims

        #batch normalization features
        self.bn_fts = nn.BatchNorm1d(self.input_dims)

        self.edge_convs = nn.ModuleList()

        # First layer: from input_dims to conv_params[0]
        nns = nn.Sequential(
            nn.Linear(self.input_dims, conv_params[0]),
            nn.BatchNorm1d(conv_params[0]),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(conv_params[0], conv_params[0]),
            nn.ReLU(),
            nn.BatchNorm1d(conv_params[0]),
            nn.Dropout(0.6)
        )
        self.edge_convs.append(GINEConv(nn=nns, edge_dim=num_edge_features))

        # Remaining layers
        for in_feat, out_feat in zip(conv_params[:-1], conv_params[1:]):
            nns = nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.Dropout(0.6),
                nn.ReLU(),
                nn.Linear(out_feat, out_feat),
                nn.ReLU(),
                nn.BatchNorm1d(out_feat),
                nn.Dropout(0.6)
            )
            self.edge_convs.append(GINEConv(nn=nns, edge_dim=num_edge_features))

        if self.use_fusion:
            in_chn = sum(x for x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Linear(in_chn, out_chn), nn.ReLU(), nn.BatchNorm1d(out_chn))

        fcs = []
        for idx, (channels, drop_rate) in enumerate(fc_params):
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1]  
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
            nn.BatchNorm1d(num_classes)
        ))

        self.fc = nn.Sequential(*fcs)

    def forward(self, data):
    
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #batch normalization features
        x = self.bn_fts(x)
        
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            x = conv(x, edge_index, edge_attr)
            if self.use_fusion:
                outputs.append(x)
        #fuse outputs from all GINEConv layers
        if self.use_fusion:
            x = self.fusion_block(torch.cat(outputs, dim=1))

        #Global average pooling
        x = global_mean_pool(x, batch = data.batch) 
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x