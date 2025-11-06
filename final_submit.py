import os
import math
import json
import copy
import random
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdchem
from rdkit.Chem.EState import Fingerprinter as EStateFingerprinter
from rdkit.Chem import rdPartialCharges as RPC
from rdkit import DataStructs as RDDataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
    GlobalAttention,
    GraphNorm,
)
from torch_geometric.utils import dropout_adj

# ------------------------
# 实用函数
# ------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 常见元素 + 星号（原子号0）
AN_LIST = [0, 1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
AN2IDX = {z: i for i, z in enumerate(AN_LIST)}

HYB_LIST = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2]

# 子结构 SMARTS（少量稳健片段）
SMARTS = {
    'carbonyl': Chem.MolFromSmarts('[CX3]=[OX1]'),
    'ester': Chem.MolFromSmarts('C(=O)O'),
    'amide': Chem.MolFromSmarts('C(=O)N'),
    'ether': Chem.MolFromSmarts('COC'),
    'aromaticN': Chem.MolFromSmarts('n'),
}


def mol_from_cru(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    return mol


def atom_feature_vec(atom: rdchem.Atom) -> List[float]:
    Z = atom.GetAtomicNum()
    # 1) 原子号 one-hot（含 * =0）+ 其他桶
    an = [0.0] * (len(AN_LIST) + 1)
    an[AN2IDX.get(Z, len(AN_LIST))] = 1.0
    # 2) 度（0..5）
    deg = [0.0] * 6
    deg[min(atom.GetTotalDegree(), 5)] = 1.0
    # 3) 形式电荷（-2..+2）
    fc = [0.0] * 5
    ch = int(np.clip(atom.GetFormalCharge(), -2, 2))
    fc[ch + 2] = 1.0
    # 4) 杂化
    hyb = [0.0] * (len(HYB_LIST) + 1)
    h = atom.GetHybridization()
    idx = HYB_LIST.index(h) if h in HYB_LIST else len(HYB_LIST)
    hyb[idx] = 1.0
    # 5) 其他布尔特征
    flags = [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(Z == 0),  # is_attachment
    ]
    # 6) 总氢（0..4）
    th = [0.0] * 5
    th[min(atom.GetTotalNumHs(includeNeighbors=True), 4)] = 1.0
    # 7) 近似相对原子质量（缩放）
    mass = [float(atom.GetMass() / 200.0)]
    return an + deg + fc + hyb + flags + th + mass


BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]


def bond_feature_vec(bond: rdchem.Bond) -> List[float]:
    bt = [0.0] * (len(BOND_TYPES) + 1)
    btype = bond.GetBondType()
    idx = BOND_TYPES.index(btype) if btype in BOND_TYPES else len(BOND_TYPES)
    bt[idx] = 1.0
    flags = [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    # 立体：NONE/E/Z/OTHER
    stereo_map = {
        rdchem.BondStereo.STEREONONE: 0,
        rdchem.BondStereo.STEREOE: 1,
        rdchem.BondStereo.STEREOZ: 2,
    }
    s = [0.0] * 4
    s[stereo_map.get(bond.GetStereo(), 3)] = 1.0
    return bt + flags + s


# 连接点（*）相关：最短路长度（去掉 *-邻接的两条边）
from collections import deque

def shortest_path_len_excluding_star_edges(mol: Chem.Mol) -> float:
    star_ids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(star_ids) < 2:
        return -1.0
    s, t = star_ids[0], star_ids[1]
    # BFS，但路径长度剔除两侧 *-邻接边（相当于 -2）
    g = [[] for _ in range(mol.GetNumAtoms())]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        g[i].append(j)
        g[j].append(i)
    # 标准 BFS
    q = deque([s])
    dist = [-1] * mol.GetNumAtoms()
    dist[s] = 0
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in g[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    d = dist[t]
    if d == -1:
        return -1.0
    # 剔除两端 *-邻接边的贡献
    d = max(0, d - 2)
    return float(d)


def _gasteiger_stats(mol: Chem.Mol):
    mean_q, std_q = 0.0, 0.0
    try:
        RPC.ComputeGasteigerCharges(mol)
        charges = []
        for a in mol.GetAtoms():
            if a.HasProp('_GasteigerCharge'):
                try:
                    charges.append(float(a.GetProp('_GasteigerCharge')))
                except Exception:
                    pass
        if len(charges):
            arr = np.nan_to_num(np.asarray(charges, dtype=np.float32))
            mean_q = float(arr.mean())
            std_q = float(arr.std())
    except Exception:
        pass
    return mean_q, std_q


def global_features(mol: Chem.Mol) -> np.ndarray:
    # —— 小工具 —— #
    def _safe_float(x, default=0.0):
        try:
            v = float(x)
            if not np.isfinite(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    def _safe_div(a, b, default=0.0):
        b = float(b)
        if not np.isfinite(b) or abs(b) < 1e-12:
            return float(default)
        v = float(a) / b
        return v if np.isfinite(v) else float(default)

    # 基础计数
    star_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (0, 1, 6))
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings() if ring_info is not None else 0
    arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    try:
        rot_bonds = Descriptors.NumRotatableBonds(mol)
    except Exception:
        rot_bonds = 0

    # RDKit 描述符（全部 try/except + 安全转 float）
    try:
        tpsa = _safe_float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        tpsa = 0.0
    try:
        logp = _safe_float(Descriptors.MolLogP(mol))
    except Exception:
        logp = 0.0
    try:
        molwt = _safe_float(Descriptors.MolWt(mol))
    except Exception:
        molwt = 0.0
    try:
        hbd = _safe_float(Descriptors.NumHDonors(mol))
    except Exception:
        hbd = 0.0
    try:
        hba = _safe_float(Descriptors.NumHAcceptors(mol))
    except Exception:
        hba = 0.0
    try:
        fcsp3_raw = Descriptors.FractionCSP3(mol) if hasattr(Descriptors, 'FractionCSP3') else 0.0
        fcsp3 = _safe_float(fcsp3_raw)
    except Exception:
        fcsp3 = 0.0

    # Gasteiger 电荷统计（已做 nan→0）
    q_mean, q_std = _gasteiger_stats(mol)

    # 子结构计数
    sub_counts = []
    for name, patt in SMARTS.items():
        try:
            sub_counts.append(len(mol.GetSubstructMatches(patt)))
        except Exception:
            sub_counts.append(0)

    # EState 汇总
    try:
        estate_vec, _ = EStateFingerprinter(mol)
        estate_arr = np.asarray(estate_vec, dtype=np.float64)
        estate_arr = np.nan_to_num(estate_arr, nan=0.0, posinf=0.0, neginf=0.0)
        estate_sum = float(estate_arr.sum())
        estate_nz = float(np.count_nonzero(estate_arr))
        estate_mean = float(estate_arr.mean()) if estate_arr.size else 0.0
    except Exception:
        estate_sum = estate_nz = estate_mean = 0.0

    sp_len = shortest_path_len_excluding_star_edges(mol)

    # 比例型特征用安全除法
    hetero_ratio = _safe_div(hetero, max(1, heavy_atoms), default=0.0)
    arom_ratio = _safe_div(arom_atoms, max(1, heavy_atoms), default=0.0)

    feats = [
        float(star_count),
        float(heavy_atoms),
        float(hetero),
        float(num_rings),
        float(arom_atoms),
        float(rot_bonds),
        float(sp_len),
        float(hetero_ratio),
        float(arom_ratio),
        # RDKit 描述符
        tpsa, logp, molwt, hbd, hba, fcsp3,
        float(q_mean), float(q_std),
        float(estate_sum), float(estate_nz), float(estate_mean),
    ] + [float(x) for x in sub_counts]

    # —— 最终消毒 + 限幅（防止极端值炸 float32 / scaler） —— #
    arr = np.asarray(feats, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    arr = np.clip(arr, -1e6, 1e6)
    return arr.astype(np.float32)


@dataclass
class GraphExample:
    data: Data
    y: np.ndarray  # (P,)
    y_mask: np.ndarray  # (P,) 1=present, 0=missing


class CRUDataset(Dataset):
    def __init__(self, df: pd.DataFrame, smiles_col: str, target_cols: List[str], scaler_dict=None):
        super().__init__()
        self.target_cols = target_cols
        self.examples: List[GraphExample] = []
        self.global_scaler = StandardScaler() if scaler_dict is None else scaler_dict['global']
        self._build(df, smiles_col, target_cols, fit_scaler=(scaler_dict is None))

    def _build(self, df: pd.DataFrame, smiles_col: str, target_cols: List[str], fit_scaler: bool):
        gfeats, ys, ymasks, datas = [], [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Building graphs'):
            smi = str(row[smiles_col])
            mol = mol_from_cru(smi)
            if mol is None:
                continue
            # 节点
            x = [atom_feature_vec(a) for a in mol.GetAtoms()]
            x = torch.tensor(np.asarray(x, dtype=np.float32))
            # 边
            edge_index = [[], []]
            eattr = []
            for bond in mol.GetBonds():
                i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = bond_feature_vec(bond)
                edge_index[0].extend([i1, i2])
                edge_index[1].extend([i2, i1])
                eattr.append(bf)
                eattr.append(bf)
            if len(eattr) == 0:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                eattr = torch.zeros((1, (len(BOND_TYPES)+1) + 2 + 4), dtype=torch.float32)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                eattr = torch.tensor(np.asarray(eattr, dtype=np.float32))
            # 全局特征
            gf = global_features(mol)
            gfeats.append(gf)
            # 目标与掩码
            y = []
            m = []
            for c in target_cols:
                val = row.get(c, np.nan)
                if pd.isna(val):
                    y.append(0.0); m.append(0.0)
                else:
                    y.append(float(val)); m.append(1.0)
            y = np.asarray(y, dtype=np.float32)
            m = np.asarray(m, dtype=np.float32)
            data = Data(x=x, edge_index=edge_index, edge_attr=eattr)
            ys.append(y); ymasks.append(m); datas.append(data)
    

        gfeats = np.vstack(gfeats) if len(gfeats) else np.zeros((0, 1), dtype=np.float32)

        # —— 汇总级别再消毒：去 NaN/Inf + 限幅，确保传给 scaler 的全是有限数 —— #
        if gfeats.size > 0:
            gfeats = gfeats.astype(np.float64, copy=False)
            gfeats = np.nan_to_num(gfeats, nan=0.0, posinf=1e6, neginf=-1e6)
            gfeats = np.clip(gfeats, -1e6, 1e6).astype(np.float32, copy=False)

        if fit_scaler and gfeats.size > 0:
            self.global_scaler.fit(gfeats)
        if gfeats.size > 0:
            gfeats = self.global_scaler.transform(gfeats)

        # 绑定回 Data
        for data, gf, y, m in zip(datas, gfeats, ys, ymasks):
            data.global_x = torch.tensor(gf, dtype=torch.float32).unsqueeze(0)
            self.examples.append(GraphExample(data=data, y=y, y_mask=m))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        data = ex.data
        data.y = torch.tensor(ex.y, dtype=torch.float32).unsqueeze(0)
        data.y_mask = torch.tensor(ex.y_mask, dtype=torch.float32).unsqueeze(0)
        return data


# ------------------------
# ECFP + Butina 聚类拆分（避免泄漏）
# ------------------------
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Cluster import Butina

STAR_AWARE_TAG = 999999

def star_aware_ecfp(cru_smi: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    m = Chem.MolFromSmiles(cru_smi, sanitize=True)
    if m is None:
        return np.zeros((nBits,), dtype=np.int8)
    try:
        base_inv = list(rdMolDescriptors.GetConnectivityInvariants(m))
        inv = [(STAR_AWARE_TAG if m.GetAtomWithIdx(i).GetAtomicNum() == 0 else base_inv[i])
               for i in range(m.GetNumAtoms())]
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits, invariants=inv)
        arr = np.zeros((nBits,), dtype=np.int8)
        RDDataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros((nBits,), dtype=np.int8)


def tanimoto_sim(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def butina_cluster(fps: List[np.ndarray], thresh: float = 0.6) -> List[List[int]]:
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = [tanimoto_sim(fps[i], fps[j]) for j in range(i)]
        dists.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dists, nPts=n, distThresh=1 - thresh, isDistData=True)
    return [list(c) for c in clusters]


def clustered_kfold_indices(smiles_list: List[str], k: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    fps = [star_aware_ecfp(s) for s in tqdm(smiles_list, desc='ECFP for split (star-aware)')]
    clusters = butina_cluster(fps, thresh=0.6)
    rng = random.Random(seed)
    rng.shuffle(clusters)
    folds = [[] for _ in range(k)]
    for i, cl in enumerate(clusters):
        folds[i % k].extend(cl)
    indices = np.arange(len(smiles_list))
    splits = []
    for i in range(k):
        val_idx = np.array(sorted(folds[i]))
        train_idx = np.array(sorted(np.setdiff1d(indices, val_idx)))
        splits.append((train_idx, val_idx))
    return splits


# ------------------------
# 模型定义
# ------------------------
class GNNBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(self.mlp, edge_dim=edge_dim)
        self.norm = GraphNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class CRUGNN(nn.Module):
    def __init__(self, node_in: int, edge_in: int, global_in: int, hidden: int, num_layers: int, out_dim: int,
                 dropout: float = 0.1, edge_drop_p: float = 0.05):
        super().__init__()
        self.edge_drop_p = float(edge_drop_p)
        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_embed = nn.Linear(edge_in, hidden)
        self.layers = nn.ModuleList([GNNBlock(hidden, hidden, dropout=dropout) for _ in range(num_layers)])
        # 池化：均值 + 注意力
        self.gate_nn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.global_attn = GlobalAttention(self.gate_nn)
        readout_dim = hidden * 2
        self.global_proj = nn.Linear(global_in, hidden)
        fusion_dim = readout_dim + hidden
        self.head_pre_norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.node_embed(x))
        edge_attr = F.relu(self.edge_embed(edge_attr))
        # EdgeDrop（训练期）
        if self.training and self.edge_drop_p > 0.0:
            edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.edge_drop_p, force_undirected=False,
                                                num_nodes=x.size(0), training=True)
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)
        h_mean = global_mean_pool(x, batch)
        h_attn = self.global_attn(x, batch)
        h_graph = torch.cat([h_mean, h_attn], dim=-1)
        g = F.relu(self.global_proj(data.global_x))
        h = torch.cat([h_graph, g], dim=-1)
        h = self.head_pre_norm(h)
        out = self.head(h)
        return out


# ------------------------
# 训练与评估
# ------------------------

def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum(dim=0).clamp(min=1.0)
    task_mse = diff.sum(dim=0) / denom
    loss = task_mse.mean()
    return loss, task_mse


def masked_l1_weighted(pred, target, mask, weight_z: torch.Tensor, alpha: float = 0.8):
    """按任务加权的 L1（z 空间）+ 少量 MSE 稳定训练。
    weight_z = w_i * sd_i 近似物理空间的 wMAE 权重。
    返回标量 loss。
    """
    diff = (pred - target).abs() * mask
    denom = mask.sum(dim=0).clamp(min=1.0)
    task_mae_z = diff.sum(dim=0) / denom  # (P,)
    l1_w = (task_mae_z * weight_z).mean()
    # 少量 MSE 稳定
    mse = (( (pred - target) ** 2 * mask ).sum(dim=0) / denom).mean()
    return alpha * l1_w + (1.0 - alpha) * mse


def zscore_fit(y: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.zeros(y.shape[1], dtype=np.float32)
    sd = np.ones(y.shape[1], dtype=np.float32)
    for p in range(y.shape[1]):
        vals = y[mask[:, p] > 0.5, p]
        if len(vals) >= 2:
            mu[p] = float(vals.mean())
            s = float(vals.std(ddof=0))
            sd[p] = s if s > 1e-8 else 1.0
    return mu, sd


def zscore_apply(y: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor):
    return (y - mu) / sd


def inverse_zscore(y: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return y * sd + mu

# ---- wMAE 权重计算（基于训练集）----
def compute_wmae_weights(df: pd.DataFrame, targets: List[str]) -> np.ndarray:
    """
   w_i = (1/r_i) * (K * sqrt(1/n_i) / sum_j sqrt(1/n_j))
    - n_i: 该属性的有效样本数（非空）
    - r_i: 该属性在训练集上的取值范围 max-min
    """
    K = len(targets)
    ni = np.array([df[t].notna().sum() for t in targets], dtype=np.float64)
    ranges = []
    for t in targets:
        vals = df[t].dropna().values
        ranges.append(float(np.max(vals) - np.min(vals)) if len(vals) else 0.0)
    ri = np.asarray(ranges, dtype=np.float64)
    sqrt_inv_n = np.sqrt(1.0 / np.clip(ni, 1.0, None))
    norm = (K * sqrt_inv_n) / max(1e-12, sqrt_inv_n.sum())
    inv_r = 1.0 / np.clip(ri, 1e-12, None)
    w = inv_r * norm
    return w.astype(np.float64)


@dataclass
class TrainConfig:
    csv: str
    smiles_col: str
    targets: List[str]
    epochs: int = 120
    batch_size: int = 64
    lr: float = 3e-4
    hidden_dim: int = 256
    layers: int = 5
    dropout: float = 0.1
    folds: int = 5
    split_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Early Stopping（按 wMAE）
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.0
    # EdgeDrop 概率
    edge_drop_p: float = 0.05
    # L1/MSE 混合
    alpha_l1: float = 0.8
    # EMA / SWA
    use_ema: bool = True
    ema_decay: float = 0.995
    use_swa: bool = False
    # 输出目录（Kaggle 下固定到 /kaggle/working/checkpoints）
    out_dir: str = '/kaggle/working/checkpoints'



def _make_scheduler_with_warmup(optim, total_epochs: int):
    warmup = max(3, total_epochs // 20)
    def lr_lambda(e):
        # e 从 1 开始传入，这里做个保护
        ee = max(1, e)
        if ee <= warmup:
            return ee / float(max(1, warmup))
        progress = (ee - warmup) / float(max(1, total_epochs - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=(1.0 - d))
        for (n_ema, b_ema), (n, b) in zip(self.ema.named_buffers(), model.named_buffers()):
            b_ema.data.copy_(b.data)


def run_fold(cfg: TrainConfig, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, fold_id: int):
    set_seed(1234 + fold_id)
    # 构建数据集与 z-score（只在训练集上拟合）
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # 先构造一次训练集以拟合 global_scaler
    train_tmp = CRUDataset(train_df, cfg.smiles_col, cfg.targets)
    scaler_dict = {'global': train_tmp.global_scaler}
    # 重新构造 train/val（共享 scaler）
    train_ds = CRUDataset(train_df, cfg.smiles_col, cfg.targets, scaler_dict=scaler_dict)
    val_ds = CRUDataset(val_df, cfg.smiles_col, cfg.targets, scaler_dict=scaler_dict)

    # 统计 z-score 参数（在训练集标签上）
    y_stack = np.stack([ex.y for ex in train_ds.examples])
    m_stack = np.stack([ex.y_mask for ex in train_ds.examples])
    mu, sd = zscore_fit(y_stack, m_stack)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=cfg.device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=cfg.device)
    # 计算 wMAE 的属性权重（在训练集上）并转到设备
    w = compute_wmae_weights(train_df, cfg.targets)
    weight_z = torch.tensor(w * sd, dtype=torch.float32, device=cfg.device)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    # 维度
    sample = train_ds.examples[0].data
    node_in = sample.x.size(-1)
    edge_in = sample.edge_attr.size(-1)
    global_in = sample.global_x.size(-1)

    model = CRUGNN(node_in, edge_in, global_in, cfg.hidden_dim, cfg.layers, out_dim=len(cfg.targets),
                   dropout=cfg.dropout, edge_drop_p=cfg.edge_drop_p).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = _make_scheduler_with_warmup(optim, total_epochs=cfg.epochs)

    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Fold {fold_id} Epoch {epoch}/{cfg.epochs} [train]'):
            batch = batch.to(cfg.device)
            pred = model(batch)
            y = batch.y
            m = batch.y_mask
            yz = zscore_apply(y, mu_t, sd_t)
            loss = masked_l1_weighted(pred, yz, m, weight_z, alpha=cfg.alpha_l1)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optim.step()
            if ema is not None:
                ema.update(model)
            train_loss += loss.item() * batch.num_graphs
        scheduler.step()
        train_loss /= max(1, len(train_loader.dataset))

        # 验证：按 wMAE 评估；若启用 EMA 用其参数
        eval_model = ema.ema if ema is not None else model
        eval_model.eval()
        with torch.no_grad():
            val_mse_list = []
            val_mae_phys = np.zeros(len(cfg.targets), dtype=np.float64)
            denom = np.zeros(len(cfg.targets), dtype=np.float64)
            wmae_sum = 0.0
            n_samples = 0
            for batch in tqdm(val_loader, desc=f'Fold {fold_id} Epoch {epoch}/{cfg.epochs} [val]'):
                batch = batch.to(cfg.device)
                pred = eval_model(batch)
                y = batch.y
                m = batch.y_mask
                yz = zscore_apply(y, mu_t, sd_t)
                _, task_mse = masked_mse(pred, yz, m)
                val_mse_list.append(task_mse.detach().cpu().numpy())
                # 物理空间 MAE（逐任务，用于日志）
                pred_phys = inverse_zscore(pred.detach().cpu().numpy(), mu, sd)
                y_phys = y.detach().cpu().numpy()
                m_np = m.detach().cpu().numpy()
                err = np.abs(pred_phys - y_phys) * m_np
                val_mae_phys += err.sum(axis=0)
                denom += m_np.sum(axis=0)
                # wMAE：样本加权平均
                w_row = (err * w[None, :]).sum(axis=1)
                wmae_sum += float(w_row.sum())
                n_samples += m_np.shape[0]
            val_mse = np.mean(np.stack(val_mse_list, axis=0), axis=0)
            val_mae = val_mae_phys / np.clip(denom, 1.0, None)
            val_mse_mean = val_mse.mean()
            val_wmae = wmae_sum / max(1, n_samples)

        # Model selection & Early Stopping（统一按 wMAE）
        is_better = (val_wmae + cfg.early_stop_min_delta) < best_val
        if is_better:
            best_val = val_wmae
            state_dict = (ema.ema.state_dict() if ema is not None else model.state_dict())
            # 在 kfold_train() 里，构建 best_state 时：
                # 在 kfold_train() 里，构建 best_state 时：
            best_state = {
                'model': model.state_dict(),
                # 这两项原来是 np.ndarray，改成 list
                'mu': mu.tolist(),
                'sd': sd.tolist(),
                # StandardScaler 统计量也转成 list
                'scaler_global_mean': train_ds.global_scaler.mean_.astype(np.float32).tolist(),
                'scaler_global_scale': train_ds.global_scaler.scale_.astype(np.float32).tolist(),
                'targets': cfg.targets,
            }
            no_improve = 0
        else:
            no_improve += 1

        # 日志
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Fold {fold_id}] Epoch {epoch:03d} | train_loss(z)={train_loss:.4f} "
                  f"| wMAE={val_wmae:.6f} | val_mse(z)={val_mse_mean:.4f} "
                  f"| val_MAE_phys per task={np.round(val_mae, 6)} | best_wMAE={best_val:.6f} | no_improve={no_improve}")

        # 触发早停
        if no_improve >= cfg.early_stop_patience:
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch} ")
            break

    return best_state, best_val


def kfold_train(cfg: TrainConfig):
    df = pd.read_csv(cfg.csv)
    assert cfg.smiles_col in df.columns, f"SMILES列 {cfg.smiles_col} 不存在"
    for c in cfg.targets:
        assert c in df.columns, f"目标列 {c} 不存在"
    smiles_list = df[cfg.smiles_col].astype(str).tolist()
    splits = clustered_kfold_indices(smiles_list, k=cfg.folds, seed=cfg.split_seed)
    fold_states = []
    fold_scores = []
    os.makedirs(cfg.out_dir, exist_ok=True)
    for i, (tr, va) in enumerate(splits):
        state, score = run_fold(cfg, df, tr, va, fold_id=i)
        fold_states.append(state)
        fold_scores.append(score)
        # 保存每折最优
        if state is not None:
            ckpt_i = os.path.join(cfg.out_dir, f'cru_gnn_best_fold{i}.pt')
            torch.save(state, ckpt_i)
            print(f"Saved fold {i} checkpoint to {ckpt_i}")
    print("\n==== K-Fold Summary ====")
    for i, s in enumerate(fold_scores):
        print(f"Fold {i}: wMAE={s:.6f}")
    print(f"Mean wMAE={np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    # 保存最优折模型信息与 targets
    best_i = int(np.argmin(fold_scores))
    best = fold_states[best_i]
    ckpt_best = os.path.join(cfg.out_dir, f'cru_gnn_best_fold{best_i}.pt')
    torch.save(best, ckpt_best)
    with open(os.path.join(cfg.out_dir, 'targets.json'), 'w', encoding='utf-8') as f:
        json.dump({'targets': cfg.targets}, f, ensure_ascii=False, indent=2)
    print(f"Saved best fold ({best_i}) checkpoint to checkpoints/cru_gnn_best_fold{best_i}.pt")


# ====== Inference helpers ======
from sklearn.preprocessing import StandardScaler

def _scaler_from_state(state) -> StandardScaler:
    scaler = StandardScaler()
    mean = np.asarray(state['scaler_global_mean'], dtype=np.float32)
    scale = np.asarray(state['scaler_global_scale'], dtype=np.float32)
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = mean.shape[0]
    return scaler


def load_model_for_infer(ckpt_path: str, hidden_dim: int = 256, layers: int = 5, dropout: float = 0.1, device: str = 'cpu'):
    state = torch.load(ckpt_path, map_location=device)
    targets = state['targets']
    # 构造一个假的 sample 以恢复维度
    node_in = len(atom_feature_vec(Chem.MolFromSmiles('C').GetAtomWithIdx(0)))
    edge_in = len(bond_feature_vec(Chem.MolFromSmiles('CC').GetBondWithIdx(0)))
    global_in = len(global_features(Chem.MolFromSmiles('C')))
    model = CRUGNN(node_in, edge_in, global_in, hidden_dim, layers, out_dim=len(targets), dropout=dropout)
    model.load_state_dict(state['model'])
    model.to(device).eval()
    mu = np.asarray(state['mu'], dtype=np.float32)
    sd = np.asarray(state['sd'], dtype=np.float32)
    return model, targets, mu, sd


def predict_single(ckpt_path: str, df: pd.DataFrame, smiles_col: str, batch_size: int, device: str,
                   hidden_dim: int, layers: int, dropout: float) -> Tuple[np.ndarray, List[str]]:
    state = torch.load(ckpt_path, map_location=device)
    model, targets, mu, sd = load_model_for_infer(
        ckpt_path, hidden_dim=hidden_dim, layers=layers, dropout=dropout, device=device
    )
    scaler = _scaler_from_state(state)
    test_ds = CRUDataset(df, smiles_col, targets, scaler_dict={'global': scaler})
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_z = model(batch)
            pred_phys = inverse_zscore(pred_z.detach().cpu().numpy(), mu, sd)
            preds.append(pred_phys)
    y_pred = np.vstack(preds) if preds else np.zeros((0, len(targets)))
    return y_pred, targets


def predict_and_make_submission(
    ckpt_path: str,
    test_csv: str,
    sample_sub_csv: str,
    smiles_col: str = 'SMILES',
    batch_size: int = 256,
    hidden_dim: int = 256,
    layers: int = 5,
    dropout: float = 0.1,
    device: str = 'cpu',
    out_path: str = 'submission.csv',
):
    df_test = pd.read_csv(test_csv)
    y_pred, targets = predict_single(ckpt_path, df_test, smiles_col, batch_size, device, hidden_dim, layers, dropout)
    sub = pd.read_csv(sample_sub_csv)
    for i, col in enumerate(targets):
        sub[col] = y_pred[:, i]
    sub.to_csv(out_path, index=False)
    print(f"[submit] 保存到：{out_path}")


def predict_ensemble_and_make_submission(
    ckpt_paths: List[str],
    test_csv: str,
    sample_sub_csv: str,
    smiles_col: str = 'SMILES',
    batch_size: int = 256,
    hidden_dim: int = 256,
    layers: int = 5,
    dropout: float = 0.1,
    device: str = 'cpu',
    out_path: str = 'submission.csv',
):
    df_test = pd.read_csv(test_csv)
    all_preds = []
    targets_ref = None
    for p in ckpt_paths:
        y_pred, targets = predict_single(p, df_test, smiles_col, batch_size, device, hidden_dim, layers, dropout)
        all_preds.append(y_pred)
        if targets_ref is None:
            targets_ref = targets
    y_pred = np.mean(np.stack(all_preds, axis=0), axis=0)
    sub = pd.read_csv(sample_sub_csv)
    for i, col in enumerate(targets_ref):
        sub[col] = y_pred[:, i]
    sub.to_csv(out_path, index=False)
    print(f"[submit-ens] 保存到：{out_path}，集成 {len(ckpt_paths)} 个折模型")

# ------------------------
# Kaggle I/O auto detection
# ------------------------
def auto_detect_kaggle_paths():
    """
    在 Kaggle 环境下自动查找 train/test/sample_submission CSV。
    返回: (train_csv, test_csv, sample_sub_csv)
    若未找到则返回空字符串。
    """
    roots = ['/kaggle/input/neurips-open-polymer-prediction-2025', '/kaggle/working', '.']
    def _find(name):
        for r in roots:
            hits = glob.glob(os.path.join(r, '**', name), recursive=True)
            if len(hits):
                return hits[0]
        return ''
    train_csv = _find('train.csv')
    test_csv = _find('test.csv')
    sample_csv = _find('sample_submission.csv')
    return train_csv, test_csv, sample_csv

# ------------------------
# CLI
# ------------------------
if __name__ == '__main__':
    # 仅支持 Kaggle：自动配置与运行
    if not os.path.exists('/kaggle/input'):
        raise EnvironmentError("本脚本为 Kaggle Notebook 专用，未检测到 /kaggle/input。")

    print("[env] Kaggle environment detected.")
    # 自动探测 train/test/sample_submission 路径
    train_csv, test_csv, sample_csv = auto_detect_kaggle_paths()
    if not train_csv:
        raise FileNotFoundError("未在 /kaggle/input/** 下找到 train.csv，请将数据集添加到 Notebook。")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Kaggle 友好默认（可按需要在 Notebook 顶部改动）
    cfg = TrainConfig(
        csv=train_csv,
        smiles_col='SMILES',
        targets=['Tg','FFV','Tc','Density','Rg'],
        epochs=120,                          # 如需更快调参，可改为 80
        batch_size=(128 if torch.cuda.is_available() else 64),
        lr=3e-4,
        hidden_dim=256,
        layers=5,
        dropout=0.1,
        folds=5,
        split_seed=42,
        device=device,
        early_stop_patience=15,              # Kaggle 常见限时，略收紧早停
        early_stop_min_delta=0.0,
        edge_drop_p=0.05,
        alpha_l1=0.8,
        use_ema=True,
        ema_decay=0.995,
        use_swa=False,
        out_dir='/kaggle/working/checkpoints',
    )

    print(cfg)
    kfold_train(cfg)

    # ===== 训练完自动提交（若 test & sample_submission 存在）=====
    if test_csv and sample_csv:
        os.makedirs(cfg.out_dir, exist_ok=True)
        cands = sorted(glob.glob(os.path.join(cfg.out_dir, 'cru_gnn_best_fold*.pt')))
        if len(cands) >= 2:
            out_submit = '/kaggle/working/submission.csv'
            predict_ensemble_and_make_submission(
                ckpt_paths=cands,
                test_csv=test_csv,
                sample_sub_csv=sample_csv,
                smiles_col=cfg.smiles_col,
                batch_size=max(128, cfg.batch_size),
                hidden_dim=cfg.hidden_dim,
                layers=cfg.layers,
                dropout=cfg.dropout,
                device=cfg.device,
                out_path=out_submit,
            )
        elif len(cands) == 1:
            out_submit = '/kaggle/working/submission.csv'
            predict_and_make_submission(
                ckpt_path=cands[0],
                test_csv=test_csv,
                sample_sub_csv=sample_csv,
                smiles_col=cfg.smiles_col,
                batch_size=max(128, cfg.batch_size),
                hidden_dim=cfg.hidden_dim,
                layers=cfg.layers,
                dropout=cfg.dropout,
                device=cfg.device,
                out_path=out_submit,
            )
        else:
            print("[submit] 未找到可用 ckpt，跳过生成提交。")
    else:
        print("[submit] 未检测到 test.csv 或 sample_submission.csv，跳过生成提交。")
