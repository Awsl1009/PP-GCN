import os
import time
import torch
import shutil
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
from model.fusion import VisualTextFusion
# from model.fusion import VisualTextFusion_MLP
# from model.fusion import VisualTextFusion_Attention
from model.ctrgcn import Model_4part_ForFusion
import clip
from text.Text_Prompt import text_prompt_overall_description_clip
from feeders.feeder_CTRGCN import Feeder
from collections import OrderedDict


device = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数
batch_size = 64
test_batch_size = 128
num_epochs = 100
learning_rate = 1e-4
weight_decay = 5e-2
work_dir = './work_dir/real_fusion_model_43'
os.makedirs(work_dir, exist_ok=True)


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Fusion Training for Skeleton + Text')
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--model_path', default='./work_dir/d_gait/CTRGCN_Text_100epoch_ViT14_1.0_128/runs-65-18460.pt',
                        help='Path to pre-trained skeleton model weights')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducibility')
    return parser


# 初始化随机种子
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_dataloader(args):
    train_feeder = Feeder(**{
        'data_path': '/root/DGait_CTRGCN/data/processed_data_train/d_gait_train.npz',
        'sample_path': '/root/DGait_CTRGCN/data/denoised_data_train/dname.txt',
        'split': 'train',
        'random_choose': False,
        'random_shift': False,
        'random_move': False,
        'window_size': 64,
        'normalization': False,
        'p_interval': [0.5, 1],
        'vel': False,
        'bone': False,
        'debug': False
    })

    test_feeder = Feeder(**{
        'data_path': '/root/DGait_CTRGCN/data/processed_data_test/d_gait_test.npz',
        'sample_path': '/root/DGait_CTRGCN/data/denoised_data_test/dname.txt',
        'split': 'test',
        'window_size': 64,
        'p_interval': [0.95],
        'vel': False,
        'bone': False,
        'debug': False
    })

    train_loader = DataLoader(train_feeder, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_feeder, batch_size=test_batch_size, shuffle=False, num_workers=16)
    return train_loader, test_loader


# # Focal Loss Defined By Myself
# def focal_loss(inputs, targets, alpha=0.75, gamma=2, reduction="mean"):
#     """
#     Focal Loss implementation.
    
#     Args:
#         inputs: (B, C) 未经 softmax 的 logits
#         targets: (B,) 标签索引
#     """
#     ce_loss = F.cross_entropy(inputs, targets, reduction="none")
#     pt = torch.exp(-ce_loss)
#     loss = ((1 - pt) ** gamma) * ce_loss

#     if alpha is not None:
#         at = torch.where(targets.bool(), alpha, 1 - alpha)
#         loss = at * loss

#     if reduction == "mean":
#         return loss.mean()
#     elif reduction == "sum":
#         return loss.sum()
#     else:
#         return loss
    

# # LabelSmoothingCrossEntropy
# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.smoothing = smoothing

#     def forward(self, pred, target):
#         confidence = 1. - self.smoothing
#         log_probs = F.log_softmax(pred, dim=-1)
#         nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
#         nll = nll.squeeze(1)
#         smooth_loss = -log_probs.mean(dim=-1)
#         loss = confidence * nll + self.smoothing * smooth_loss
#         return loss.mean()


class Processor:
    def __init__(self, args):
        self.arg = args
        self.global_step = 0
        self.best_F1 = 0
        self.best_F1_epoch = 0
        self.best_precision = 0
        self.best_recall = 0
        self.device = device

        # 数据加载
        self.train_loader, self.test_loader = init_dataloader(args)

        # 骨架模型
        graph = 'graph.d_gait_d.Graph'
        graph_args = {'labeling_mode': 'spatial'}
        self.skeleton_model = Model_4part_ForFusion(
            num_class=2,
            num_point=17,
            in_channels=2,
            graph=graph,
            graph_args=graph_args
        ).to(device)
        self.skeleton_weights = torch.load(args.model_path, map_location=device)
        self.skeleton_weights = {k.replace('module.', ''): v for k, v in self.skeleton_weights.items()}
        self.skeleton_model.load_state_dict(self.skeleton_weights)
        self.skeleton_model.eval()
        for param in self.skeleton_model.parameters():
            param.requires_grad = False

        # CLIP 文本编码器
        self.clip_model, _ = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()
        self.clip_model = self.clip_model.float()

        # 文本特征
        text_features, _, _ = text_prompt_overall_description_clip()
        with torch.no_grad():
            self.ft = self.clip_model.encode_text(text_features.to(device)).float()

        # 融合模型
        self.N_all = 2
        self.num_pos = 17
        self.fusion_model = VisualTextFusion(text_dim=768, skeleton_dim=256, num_pos=self.num_pos, N_all=self.N_all).to(device)
        # self.fusion_model = VisualTextFusion_MLP(text_dim=768, vis_dim=256, N_all=self.N_all).to(device)
        # self.fusion_model = VisualTextFusion_Attention(text_dim=768, vis_dim=256).to(device)

        # 损失函数 & 优化器
        # Focal Loss
        # self.criterion_ce = lambda logits, labels: focal_loss(logits, labels, alpha=0.75, gamma=2, reduction="mean")

        # CrossEntropy Loss
        # self.criterion_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(self.device))

        # BCE Loss
        self.loss_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 7.0]).to(self.device))  # 抑郁样本少，给更高权重

        # Label Smoothed CrossEntropy Loss
        # self.criterion_ce = LabelSmoothingCrossEntropy(smoothing=0.1)

        self.optimizer = torch.optim.AdamW(self.fusion_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=learning_rate, steps_per_epoch=len(self.train_loader), epochs=num_epochs)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        # 日志路径
        self.save_dir = os.path.join(work_dir, 'runs')
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(logdir=self.save_dir)
        self.print_log(f'Log will be saved to: {self.save_dir}')


    def print_log(self, msg, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            msg = "[ " + localtime + " ] " + msg
        print(msg)
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            print(msg, file=f)


    def save_model(self, epoch):
        state_dict = self.fusion_model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        filename = os.path.join(self.save_dir, f'runs-{epoch + 1}-{int(self.global_step)}.pt')
        torch.save(weights, filename)
        self.print_log(f'Saved model to {filename}')


    # 保存源代码文件
    def save_source_files(self, source_files):
        """
        将指定的源代码文件复制到工作目录中，以便实验可复现。
        :param source_files: 要保存的文件路径列表
        """
        os.makedirs(self.save_dir, exist_ok=True)
        for src_file in source_files:
            if os.path.exists(src_file):
                dst_file = os.path.join(self.save_dir, os.path.basename(src_file))
                shutil.copyfile(src_file, dst_file)
                self.print_log(f'Saved source file: {src_file} -> {dst_file}')
            else:
                self.print_log(f'Warning: Source file not found: {src_file}')


    def train(self, epoch):
        self.fusion_model.train()
        total_loss = 0
        all_labels, all_preds = [], []

        process = tqdm(self.train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')
        for data, label, index in process:
            data = data.float().to(device)
            label = label.long().to(device)

            with torch.no_grad():
                fs = self.skeleton_model(data)  # (B, 256)

            batch_ft = self.ft.repeat_interleave(fs.shape[0], dim=0)  # (N_all * B, 768)
            logits, probs = self.fusion_model(batch_ft, fs)

            # # BCE Loss
            # targets_normal = (label == 0).float()
            # targets_depression = (label == 1).float()

            # loss_normal = self.loss_bce(logits[:, 0], targets_normal)
            # loss_depression = self.loss_bce(logits[:, 1], targets_depression)
            # loss = (loss_normal + loss_depression) / 2


            # BCE Loss with pos_weight
            # 构造 one-hot 标签
            targets = F.one_hot(label, num_classes=2).float()  # shape: (B, 2)
            # 使用 pos_weight 的 BCEWithLogitsLoss
            loss = self.loss_bce(logits, targets)


            # loss = self.criterion_ce(logits.view(-1, 2), label)
            
            # 梯度清零 + 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()
            # self.scheduler.step(epoch)

            pred = torch.argmax(probs, dim=1)
            total_loss += loss.item()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

        # Metrics
        avg_loss = total_loss / len(self.train_loader)
        precision = precision_score(all_labels, all_preds, zero_division=0, average='binary')
        recall = recall_score(all_labels, all_preds, zero_division=0, average='binary')
        f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary')

        self.writer.add_scalar('train/loss', avg_loss, epoch)
        self.writer.add_scalar('train/precision', precision, epoch)
        self.writer.add_scalar('train/recall', recall, epoch)
        self.writer.add_scalar('train/f1', f1, epoch)

        self.print_log(f'[Train] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')


    def eval(self, epoch, loader_name='test'):
        self.fusion_model.eval()
        all_labels, all_preds = [], []
        correct, total = 0, 0
        total_loss = 0

        wrong_file = os.path.join(self.save_dir, 'wrong.txt')
        result_file = os.path.join(self.save_dir, 'result.txt')

        with open(wrong_file, 'w') as fw, open(result_file, 'w') as fr:
            process = tqdm(self.test_loader, desc=f'Testing Epoch {epoch+1}')
            for data, label, index in process:
                data = data.float().to(device)
                label = label.long().to(device)

                with torch.no_grad():
                    fv = self.skeleton_model(data)
                    batch_ft = self.ft.repeat_interleave(fv.shape[0], dim=0)
                    logits, probs = self.fusion_model(batch_ft, fv)

                    pred = torch.argmax(probs, dim=1)
                    correct += (pred == label).sum().item()
                    total += label.size(0)

                    all_labels.extend(label.cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())

                    for i in range(len(index)):
                        fr.write(f"{pred[i].item()},{label[i].item()}\n")
                        if pred[i] != label[i]:
                            fw.write(f"{index[i]},{pred[i].item()},{label[i].item()}\n")

        # Metrics
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0, average='binary')
        recall = recall_score(all_labels, all_preds, zero_division=0, average='binary')
        f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary')

        self.writer.add_scalar('test/loss', total_loss / len(self.test_loader), epoch)
        self.writer.add_scalar('test/accuracy', accuracy, epoch)
        self.writer.add_scalar('test/precision', precision, epoch)
        self.writer.add_scalar('test/recall', recall, epoch)
        self.writer.add_scalar('test/f1', f1, epoch)

        self.print_log(f'[Test] Epoch {epoch+1} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

        if f1 > self.best_F1:
            self.best_F1 = f1
            self.best_F1_epoch = epoch + 1
            self.best_precision = precision
            self.best_recall = recall
            self.save_model(epoch)


    def start(self):
        # 保存当前使用的源码文件
        source_file = [
            '/root/DGait_CTRGCN/train_fusion.py',
            '/root/DGait_CTRGCN/model/fusion.py'
        ]
        self.save_source_files(source_file)

        self.print_log('Start training...')
        for epoch in range(num_epochs):
            self.train(epoch)
            if (epoch + 1) % self.arg.eval_interval == 0:
                self.eval(epoch)
                current_F1 = self.best_F1
                if current_F1 > getattr(self, 'current_best_F1', 0):  # 如果比之前最好还高
                    self.save_model(epoch)
                    self.current_best_F1 = current_F1  # 更新当前最佳F1值

            if (epoch + 1) % self.arg.save_interval == 0:
                pass  # 不再自动保存定期 checkpoint

        self.print_log(f'Best F1 Score: {self.best_F1:.4f} (Precision: {self.best_precision:.4f}, Recall: {self.best_recall:.4f}) at Epoch {self.best_F1_epoch}')


if __name__ == '__main__':
    args = get_parser().parse_args()
    init_seed(args.seed)
    processor = Processor(args)
    processor.start()