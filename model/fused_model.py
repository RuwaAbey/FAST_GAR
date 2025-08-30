import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.temporal_transformer import PoseFormer as TemporalPoseFormer
from model.FASTGAR import PoseFormer as SpatialPoseFormer
from model.FASTGAR import SceneFormer
from collections import OrderedDict

class FusedModel(nn.Module):
    def __init__(self, num_class=8, in_channels=3, num_person=12, num_point=17, num_head=6, graph=None, graph_args=dict(), spatial_weights=None, temporal_weights=None):
        super(FusedModel, self).__init__()
        self.spatial_zit = SpatialPoseFormer(
            in_channels=in_channels,
            num_person=num_person,
            num_point=num_point,
            num_head=num_head,
            graph=graph,
            graph_args=graph_args
        )
        self.temporal_zit = TemporalPoseFormer(
            in_channels=in_channels,
            num_person=num_person,
            num_point=num_point,
            num_head=num_head,
            graph=graph,
            graph_args=graph_args
        )
        self.group_transf = SceneFormer(
            num_class=num_class,
            num_head=num_head,
            num_person=num_person
        )
        # Attention-based gating
        self.query_fc = nn.Linear(192, 192)  # Query projection for spatial features
        self.key_fc = nn.Linear(192, 192)  # Key projection for temporal features
        self.value_fc = nn.Linear(192, 1)    # Value projection to compute gate
        nn.init.xavier_uniform_(self.query_fc.weight)
        nn.init.zeros_(self.query_fc.bias)
        nn.init.xavier_uniform_(self.key_fc.weight)
        nn.init.zeros_(self.key_fc.bias)
        nn.init.xavier_uniform_(self.value_fc.weight)
        nn.init.zeros_(self.value_fc.bias)
        # Linear layer after gated fusion
        self.fusion_fc = nn.Linear(192, 192)  # Transform fused features
        nn.init.xavier_uniform_(self.fusion_fc.weight)
        nn.init.zeros_(self.fusion_fc.bias)
        # Auxiliary classifiers
        self.spatial_fc = nn.Linear(192, num_class)
        self.temporal_fc = nn.Linear(192, num_class)
        nn.init.normal_(self.spatial_fc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.temporal_fc.weight, 0, math.sqrt(2. / num_class))

        # Load pre-trained weights for spatial and temporal branches if provided
        if spatial_weights is not None:
            spatial_state_dict = torch.load(spatial_weights)
            spatial_state_dict = OrderedDict([[k.split('module.')[-1], v] for k, v in spatial_state_dict.items()])
            self.spatial_zit.load_state_dict(spatial_state_dict)
        if temporal_weights is not None:
            temporal_state_dict = torch.load(temporal_weights)
            temporal_state_dict = OrderedDict([[k.split('module.')[-1], v] for k, v in temporal_state_dict.items()])
            self.temporal_zit.load_state_dict(temporal_state_dict)

    def forward(self, x):
        # Input: (N, C, T, V, M)
        # Spatial branch
        spatial_out = self.spatial_zit(x)  # (N, C=192, T, M)
        # Temporal branch
        temporal_out = self.temporal_zit(x)  # (N, C=192, T, M)

        # Attention-based gating
        N, C, T, M = spatial_out.shape
        # Reshape for attention: treat (T,M) as tokens
        spatial_flat = spatial_out.permute(0, 2, 3, 1).contiguous().view(N, T*M, C)  # (N, T*M, 192)
        temporal_flat = temporal_out.permute(0, 2, 3, 1).contiguous().view(N, T*M, C)  # (N, T*M, 192)
        # Compute attention scores
        query = self.query_fc(spatial_flat)  # (N, T*M, 192)
        key = self.key_fc(temporal_flat)    # (N, T*M, 192)
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(192)  # (N, T*M, T*M)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (N, T*M, T*M)
        # Compute gate
        value = self.value_fc(temporal_flat)  # (N, T*M, 1)
        gate = torch.bmm(attention_weights, value)  # (N, T*M, 1)
        gate = torch.sigmoid(gate.view(N, T, M, 1).permute(0, 3, 1, 2))  # (N, 1, T, M)
        # Compute convex combination
        fused_out = gate * spatial_out + (1 - gate) * temporal_out  # (N, 192, T, M)

        # Linear transformation after fusion
        fused_out = fused_out.permute(0, 2, 3, 1).contiguous().view(N, T*M, C)  # (N, T*M, 192)
        fused_out = F.relu(self.fusion_fc(fused_out))  # (N, T*M, 192)
        fused_out = fused_out.view(N, T, M, C).permute(0, 3, 1, 2)  # (N, 192, T, M)

        # Group transformer
        out = self.group_transf(fused_out)  # (N, num_class)

        # Auxiliary outputs
        spatial_aux = self.spatial_fc(spatial_out.mean(3).mean(2))  # (N, num_class)
        temporal_aux = self.temporal_fc(temporal_out.mean(3).mean(2))  # (N, num_class)

        return out, spatial_aux, temporal_aux