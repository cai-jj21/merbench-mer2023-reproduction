'''
Lightweight gated residual attention fusion for the MERBench reproduction.

The three modality encoders and prediction heads follow the official
Attention baseline. The fusion block adds normalized modality reliability
gates and a small cross-modal residual branch.
'''
import torch
import torch.nn as nn

from .modules.encoder import MLPEncoder, LSTMEncoder


class GatedResidualAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip

        if args.feat_type in ['utt']:
            encoder = MLPEncoder
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            encoder = LSTMEncoder
        else:
            raise ValueError(f'unsupported feat_type: {args.feat_type}')

        self.audio_encoder = encoder(args.audio_dim, hidden_dim, dropout)
        self.text_encoder = encoder(args.text_dim, hidden_dim, dropout)
        self.video_encoder = encoder(args.video_dim, hidden_dim, dropout)

        joint_dim = hidden_dim * 3
        self.attention_mlp = MLPEncoder(joint_dim, hidden_dim, dropout)
        self.fc_att = nn.Linear(hidden_dim, 3)

        # Estimate sample-wise modality reliability independently of attention.
        self.gate_mlp = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        # Preserve cross-modal interactions beyond the weighted modality sum.
        self.residual_mlp = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self.fc_out_1 = nn.Linear(hidden_dim, args.output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, args.output_dim2)

    def forward(self, batch):
        audio_hidden = self.audio_encoder(batch['audios'])
        text_hidden = self.text_encoder(batch['texts'])
        video_hidden = self.video_encoder(batch['videos'])

        joint = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1)
        attention_weights = torch.softmax(
            self.fc_att(self.attention_mlp(joint)), dim=1
        )
        reliability = torch.sigmoid(self.gate_mlp(joint))
        weights = attention_weights * reliability
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        modality_stack = torch.stack(
            [audio_hidden, text_hidden, video_hidden], dim=2
        )
        fused = torch.bmm(modality_stack, weights.unsqueeze(2)).squeeze(2)
        fused = fused + self.residual_scale * self.residual_mlp(joint)

        emos_out = self.fc_out_1(fused)
        vals_out = self.fc_out_2(fused)
        interloss = fused.new_zeros(())
        return fused, emos_out, vals_out, interloss
