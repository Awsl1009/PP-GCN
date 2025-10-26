import torch
from model.ctrgcn import Model_4part_ForFusion
from model.fusion import VisualTextFusion
from text.Text_Prompt import text_prompt_overall_description_clip
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# 1. 加载骨架模型
arg_config = {
    'num_class': 2,
    'num_point': 17,
    'in_channels': 2,
    'graph': 'graph.d_gait_d.Graph',
    'graph_args': {'labeling_mode': 'spatial'},
    'adaptive': True,
}
skeleton_model = Model_4part_ForFusion(**arg_config).to(device)
skeleton_model.eval()

# 加载权重
weights_path = '/root/DGait_CTRGCN/work_dir/d_gait/CTRGCN_Text_100epoch_ViT14_1.0_128/runs-65-18460.pt'
weights = torch.load(weights_path, map_location=device)
weights = {k.replace('module.', ''): v for k, v in weights.items()}
skeleton_model.load_state_dict(weights)

# ----------------------
# 2. 获取文本特征 ft
text_features, num_text_aug, _ = text_prompt_overall_description_clip()

# 加载 CLIP 模型
clip_model, _ = clip.load("ViT-L/14", device=device)
clip_model.eval()
clip_model = clip_model.float()  # 转换为 float32，避免 dtype 不一致问题

# 将文本 token 移动到 GPU
text_features = text_features.to(device)

# 编码文本特征
with torch.no_grad():
    ft = clip_model.encode_text(text_features).float()  # (N_all, 768)

# ----------------------
# 3. 构建 Fusion Module
N_all = ft.shape[0]
fusion_module = VisualTextFusion(text_dim=768, vis_dim=256, num_pos=17, N_all=N_all).to(device)

# ----------------------
# 4. 输入骨架数据进行推理
input_data = torch.randn(1, 2, 64, 17, 1).to(device)
with torch.no_grad():
    fv = skeleton_model(input_data).squeeze(0)  # (256, )

# ----------------------
# 5. 融合 + 分类
logits = fusion_module(ft, fv)

print("Predicted Attribute Logits:", logits)