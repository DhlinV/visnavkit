"""Compose temporal features with a trajectory prediction head."""

import torch
import torch.nn as nn

__all__ = ["ActionDecoder"]


class ActionDecoder(nn.Module):
    """Temporal encoder over per-frame vision tokens -> plan head."""

    def __init__(self, temporal_encoder, plan_head):
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.plan_head = plan_head

        if not self.plan_head.pretrained:
            self.apply(self._init_weights)

    # from: https://github.com/jchengai/planTF/blob/main/src/models/planTF/planning_model.py#L82
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) and m.weight is not None:  # adaLN uses affine-less LN
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.temporal_encoder(x)  # [B, F, feat_size] -> [B, F, feat_size] (or [B, feat_size] if reduced)
        # if temporal_encoder made a prediction for each token, convert it to a batch
        if x.dim() == 3:
            x = x.flatten(0, 1)  # [B, F, feat_size] -> [B * F, feat_size]

        plan_out = self.plan_head(x)  # [B * F, feat_size] -> [B * F, plan_out_size]
        return dict(plan=plan_out)

    def get_losses(self, preds, targets):
        plan_loss, plan_loss_debug = self.plan_head.get_losses(
            preds["plan"],
            targets["future_poses"],
        )
        loss_dict = dict(
            total=plan_loss["total"],
            reg=plan_loss["reg"],
            cls=plan_loss["cls"],
        )
        return loss_dict, plan_loss_debug
