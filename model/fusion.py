import torch
import torch.nn as nn


# 原论文Transformer融合
class VisualTextFusion(nn.Module):
    def __init__(self, text_dim=768, skeleton_dim=256, num_pos=17, N_all=2):
        super().__init__()

        # 文本特征768  -> 256骨架特征
        self.text_proj = nn.Conv1d(text_dim, skeleton_dim, kernel_size=1)
        
        # # 投影层 + 正则化
        # self.text_proj = nn.Sequential(
        #     nn.Conv1d(text_dim, vis_dim, kernel_size=1),
        #     nn.BatchNorm1d(vis_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        
        # 可学习的位置编码
        self.etext = nn.Parameter(torch.randn(1, skeleton_dim))  # 可学习文本特征
        self.eskeleton = nn.Parameter(torch.randn(1, skeleton_dim))  # 可学习骨架特征
        self.epos = nn.Parameter(torch.randn(num_pos, skeleton_dim))  # 可学习位置编码（骨架点数量17）
        self.N_all = N_all

        # 融合模块TransformerEncoder
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=skeleton_dim * 2, nhead=8),
            num_layers=3
        )

        # 分类头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(skeleton_dim * 2, skeleton_dim),
                nn.ReLU(),
                nn.Linear(skeleton_dim, 1)
            ) for _ in range(N_all)
        ])

        self.attn_query = nn.Parameter(torch.randn(1, skeleton_dim * 2))  # 可学习的查询向量


    def forward(self, ft, fs):
        """
        ft: (N_all * B, text_dim) 文本特征
        fs: (B, vis_dim) 骨架特征
        B: batch size 批次
        """
        # batch size
        B = fs.shape[0]

        # Step 1: 文本投影 + 加 etext
        phi_ft = self.text_proj(ft.unsqueeze(-1)).squeeze(-1) + self.etext  # (N_all * B, skeleton_dim)

        # Step 2: 扩展 fv 到 (num_pos, B, skeleton_dim)
        num_pos = 17
        fs = fs.unsqueeze(0).repeat(num_pos, 1, 1)  # (num_pos, B, skeleton_dim)

        # Step 3: 扩展位置编码到 (num_pos, B, skeleton_dim)
        eskeleton = self.eskeleton.unsqueeze(1).repeat(1, B, 1)  # (num_pos, B, skeleton_dim)
        epos = self.epos.unsqueeze(1).repeat(1, B, 1)  # (num_pos, B, skeleton_dim)

        # Step 4: 视觉特征 + 编码
        fs = fs + eskeleton + epos  # shape 匹配

        # Step 5: 拼接文本和视觉特征
        # 保留所有 N_all 文本分支
        phi_ft = phi_ft.view(self.N_all, B, -1)  # (N_all, B, skeleton_dim)
        # 扩展为 (num_pos, N_all, B, vis_dim)
        phi_ft_expanded = phi_ft.unsqueeze(0).repeat(num_pos, 1, 1, 1)  # (num_pos, N_all, B, skeleton_dim)
        fs_expanded = fs.unsqueeze(1).repeat(1, self.N_all, 1, 1)  # (num_pos, N_all, B, skeleton_dim)
        # 拼接
        fst = torch.cat([phi_ft_expanded, fs_expanded], dim=-1)  # (num_pos, N_all, B, 2*skeleton_dim)
        fst_seq = fst.view(num_pos, self.N_all * B, -1)  # (num_pos, B*N_all, 2*skeleton_dim)

        # Step 6: Transformer 融合
        fst_fused_seq = self.fusion(fst_seq)  # (num_pos, B*N_all, 2*skeleton_dim)

        # Step 7: 注意力加权求和
        attn_weights = torch.matmul(fst_fused_seq, self.attn_query.unsqueeze(-1))  # 计算注意力权重
        attn_weights = torch.softmax(attn_weights, dim=0)  # 归一化得到注意力分数
        global_feat = torch.sum(fst_fused_seq * attn_weights, dim=0)  # 加权求和

        # Step 8: 分类头
        logits = torch.cat([
            head(global_feat[i*B:(i+1)*B]) for i, head in enumerate(self.heads)
        ], dim=1)  # (B, N_all)

        probs = torch.softmax(logits, dim=1)  # (B, N_all)

        return logits, probs
    

# # 轻量级 MLP 融合
# class VisualTextFusion_MLP(nn.Module):
#     def __init__(self, text_dim=768, vis_dim=256, N_all=2):
#         super().__init__()
#         self.text_proj = nn.Sequential(
#             nn.Linear(text_dim, vis_dim),
#             nn.LayerNorm(vis_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )

#         # 特征融合层
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(vis_dim * 2, vis_dim),
#             nn.LayerNorm(vis_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(vis_dim, 2)
#         )

#     def forward(self, ft, fv):
#         # ft: (N_all * B, 768), fv: (B, 256)
#         phi_ft = self.text_proj(ft)  # (N_all * B, 256)
#         phi_ft = phi_ft.view(-1, fv.shape[0], phi_ft.shape[-1])  # (N_all, B, 256)

#         # 多分支融合
#         logits_list = []
#         for i in range(phi_ft.shape[0]):
#             combined = torch.cat([phi_ft[i], fv], dim=-1)  # (B, 512)
#             logits = self.fusion_layer(combined)  # (B, 2)
#             logits_list.append(logits.unsqueeze(0))

#         logits = torch.cat(logits_list, dim=0).mean(dim=0)  # 取平均
#         probs = torch.softmax(logits, dim=1)

#         return logits, probs
    

# # 加 attention 的融合
# class VisualTextFusion_Attention(nn.Module):
#     def __init__(self, text_dim=768, vis_dim=256, num_heads=4):
#         super().__init__()
#         self.text_proj = nn.Linear(text_dim, vis_dim)
#         self.attention = nn.MultiheadAttention(embed_dim=vis_dim, num_heads=num_heads)
#         self.classifier = nn.Sequential(
#             nn.Linear(vis_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 2)
#         )


#     def forward(self, ft, fv):
#         B = fv.size(0)

#         phi_ft = self.text_proj(ft)  # (N_all * B, 256)
#         phi_ft = phi_ft.view(-1, B, phi_ft.shape[-1])  # (N_all, B, 256)

#         query = fv.unsqueeze(0)  # (1, B, 256)
#         key = phi_ft
#         value = phi_ft

#         attn_output, _ = self.attention(query=query, key=key, value=value)
#         logits = self.classifier(attn_output.squeeze(0))
#         probs = torch.softmax(logits, dim=1)

#         return logits, probs