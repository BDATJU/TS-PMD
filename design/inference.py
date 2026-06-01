import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import esm
import logging
import numpy as np
from torchdrug import data
from MBSE.Stability.train import ProteinStabilityPredictor, patch_torchdrug_enhanced

patch_torchdrug_enhanced()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# =====================================================================
# 1. 全局模型初始化
# =====================================================================
log.info("⏳ 正在初始化模型...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.eval().to(DEVICE)
for p in esm_model.parameters(): p.requires_grad = False

GEARNET_PATH = "/root/BDA/algorithm-platform/algorithms/MBSE/Stability/gearnet_edge.pth"
BEST_MODEL_PATH = "/root/BDA/algorithm-platform/algorithms/MBSE/Stability/best_model.pth"

evaluator = ProteinStabilityPredictor(GEARNET_PATH, hidden_dim=128, dropout=0.5).to(DEVICE)
if os.path.exists(BEST_MODEL_PATH):
    evaluator.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
evaluator.eval()
for p in evaluator.parameters(): p.requires_grad = False

# =====================================================================
# 2. 生成器网络定义 (与 val_t7 MLP 完全一致)
# =====================================================================
class TargetConditionedMLPGenerator(nn.Module):
    def __init__(self, feature_dim=20, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, feature_dim)
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
        
    def forward(self, esm_prior_logits):
        hidden = self.relu(self.layer1(esm_prior_logits))
        delta_theta = self.layer2(hidden)
        generated_logits = esm_prior_logits + delta_theta
        return generated_logits, delta_theta

class AmortizedSequenceGenerator(nn.Module):
    def __init__(self, wt_seq, esm_model, alphabet, hidden_dim=64, device='cuda'):
        super().__init__()
        self.aa_indices = [alphabet.get_idx(tok) for tok in STANDARD_AMINO_ACIDS]
        self.esm_embed_weights = esm_model.embed_tokens.weight[self.aa_indices].detach().to(device)
        
        with torch.no_grad():
            batch_converter = alphabet.get_batch_converter()
            _, _, batch_tokens = batch_converter([("WT", wt_seq)])
            batch_tokens = batch_tokens.to(device)
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            logits = results["logits"][0, 1:-1] 
            self.esm_logits_20 = logits[:, self.aa_indices]
            self.P_esm = F.softmax(self.esm_logits_20, dim=-1).detach() 
            
        self.mlp_adapter = TargetConditionedMLPGenerator(feature_dim=20, hidden_dim=hidden_dim).to(device)
        
    def forward(self, temperature=1.0):
        combined_logits, current_delta = self.mlp_adapter(self.esm_logits_20)
        P_soft = F.gumbel_softmax(combined_logits, tau=temperature, hard=True)
        mut_seq_emb = torch.matmul(P_soft, self.esm_embed_weights)
        mut_struct_feature = F.pad(P_soft, (0, 1), value=0.0) 
        return P_soft, mut_seq_emb, mut_struct_feature, current_delta

# =====================================================================
# 3. 完美复刻的 PDB 处理与打包工具
# =====================================================================
def download_pdb(pdb_id, save_dir="./static"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if not os.path.exists(save_path):
        url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "w") as f: f.write(response.text)
    return save_path

# 🌟 将你提供的安全解析逻辑完美融入
def parse_and_build_graph_robust(pdb_path, target_chain='A', device='cuda'):
    seq, ca_coords, res_types, pdb_indices = [], [], [], []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM  "):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    chain_id = line[21]
                    if chain_id != target_chain:
                        continue
                        
                    res_name = line[17:20].strip()
                    pdb_res_num = line[22:27].strip() # 改为 string 保存插入码
                    
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    aa1 = AA3_TO_1.get(res_name, 'X')
                    seq.append(aa1)
                    ca_coords.append([x, y, z])
                    pdb_indices.append(pdb_res_num)
                    
                    if aa1 in STANDARD_AMINO_ACIDS:
                        res_types.append(STANDARD_AMINO_ACIDS.index(aa1))
                    else:
                        res_types.append(20)
                        
    coords = torch.tensor(ca_coords, dtype=torch.float32)
    res_types_t = torch.tensor(res_types, dtype=torch.long)
    num_node = len(seq)
    
    # 手动建图，彻底抛弃 TorchDrug GraphConstruction
    edge_list, bond_type = [], []
    for i in range(num_node - 1):
        edge_list.append([i, i+1, 0])
        edge_list.append([i+1, i, 0])
        bond_type.extend([0, 0])
        
    dist_matrix = torch.cdist(coords, coords)
    k = min(10, num_node - 1)
    _, knn_indices = dist_matrix.topk(k + 1, dim=-1, largest=False)
    for i in range(num_node):
        for j in knn_indices[i][1:]:
            edge_list.append([i, j.item(), 1])
            bond_type.append(1)
            
    protein = data.Protein(
        edge_list=torch.tensor(edge_list, dtype=torch.long),
        atom_type=res_types_t,         
        bond_type=torch.tensor(bond_type, dtype=torch.long),
        residue_type=res_types_t,
        num_node=num_node,
        num_relation=7               
    )
    protein.node_position = coords
    protein.atom_position = coords
    protein.atom_feature = F.one_hot(res_types_t.clamp(0, 20), num_classes=21).float()
    
    # 提取 ESM 特征
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("WT", "".join(seq))])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        seq_emb = results["representations"][33][0, 1:-1]
        
    return protein, seq_emb, "".join(seq), pdb_indices

# 🌟 100% 还原 val_t7 的打包逻辑
def prepare_packed_data(graph, seq_emb, device='cuda'):
    g = graph
    pos = getattr(g, 'node_position', None)
    if pos is None: pos = getattr(g, 'atom_position', None)
    if pos is None: pos = torch.zeros((g.num_node, 3))
    feat = getattr(g, 'atom_feature', None)
    if feat is None: feat = torch.zeros((g.num_node, 21))
    edge_f = getattr(g, 'bond_feature', None)
    if edge_f is None: edge_f = getattr(g, 'edge_feature', None)
    if edge_f is None: edge_f = torch.zeros((g.edge_list.shape[0], 59))
    res_type = getattr(g, 'residue_type', None)
    if res_type is None: res_type = torch.zeros(g.num_node, dtype=torch.long)

    skel = data.Protein(edge_list=g.edge_list, atom_type=g.atom_type, bond_type=g.bond_type, residue_type=res_type, num_node=g.num_node, num_relation=7)
    packed = data.Protein.pack([skel]).to(device)
    
    with packed.node(): 
        packed.node_position = pos.to(device)
        packed.atom_feature = feat.to(device)
        packed.node_feature = packed.atom_feature
    with packed.edge(): 
        packed.bond_feature = edge_f.to(device)
        packed.edge_feature = packed.bond_feature

    seq_pad = seq_emb.unsqueeze(0).to(device)
    mask = torch.ones(1, seq_emb.size(0), dtype=torch.bool).to(device)
    return packed, seq_pad, mask

# =====================================================================
# 4. 主设计接口
# =====================================================================
def run_protein_design(config):
    params = vars(config) if hasattr(config, '__dict__') else config
    task_id = params.get('task_id', 'unknown')
    pdb_id = params.get('pdb_id', '').lower()
    chain_id = params.get('chain_id', 'A').upper()
    top_k = int(params.get('top_k', 50)) 
    raw_steps = params.get('num_steps', params.get('steps', 1500))
    num_steps = int(raw_steps.item()) if torch.is_tensor(raw_steps) else int(raw_steps)
    
    lambda_ddg = float(params.get('lambda_ddg', 1.0))
    lambda_l1 = float(params.get('lambda_l1', 0.5))
    lambda_kl = float(params.get('lambda_kl', 0.01))
    threshold = float(params.get('threshold', 0.5))

    log.info(f"🧬 [设计任务 {task_id}] 开始处理 PDB {pdb_id}_{chain_id}")
    
    try:
        raw_pdb_path = download_pdb(pdb_id, getattr(config, 'static_dir', './static'))
        
        # 🌟 调用完美对齐的建图逻辑
        wt_raw_graph, wt_raw_esm, wt_seq, seq_idx_to_pdb_id = parse_and_build_graph_robust(raw_pdb_path, chain_id, DEVICE)
        seq_len = len(wt_seq)
        
        generator_module = AmortizedSequenceGenerator(wt_seq, esm_model, alphabet, hidden_dim=64, device=DEVICE)
        optimizer = torch.optim.Adam(generator_module.mlp_adapter.parameters(), lr=0.01)
        
        # 🌟 调用完美对齐的打包逻辑
        wt_g_packed, wt_s_pad, wt_mask = prepare_packed_data(wt_raw_graph, wt_raw_esm, DEVICE)

        log.info(f"开始推断训练 (Steps={num_steps})...")
        for step in range(num_steps):
            temp = max(0.1, 1.0 * (0.95 ** (step // 30)))
            optimizer.zero_grad()
            
            P_soft, mut_s_emb, mut_struct_feat, current_delta = generator_module(temperature=temp)
            mut_s_pad = mut_s_emb.unsqueeze(0)
            
            # 由于全长对齐，这里的图克隆绝对安全
            mut_g_fake = wt_g_packed.clone()
            with mut_g_fake.node():
                mut_g_fake.atom_feature = mut_struct_feat
                mut_g_fake.node_feature = mut_struct_feat

            random_pos = np.random.randint(0, seq_len)
            mut_pos_tensor = torch.tensor([random_pos], dtype=torch.long).to(DEVICE)

            stability_logits = evaluator(
                wt_g_packed, mut_g_fake,           
                wt_g_packed, mut_g_fake,           
                wt_s_pad, mut_s_pad,               
                wt_mask, wt_mask,                  
                mut_pos_tensor                     
            )
            
            L_ddg = -stability_logits[0, 1] 
            L_sparse = torch.abs(current_delta).mean()
            L_kl = F.kl_div(torch.log(P_soft + 1e-8), generator_module.P_esm, reduction='batchmean', log_target=False)
            
            total_loss = lambda_ddg * L_ddg + lambda_l1 * L_sparse + lambda_kl * L_kl
            total_loss.backward()
            optimizer.step()
            
            if (step + 1) % 100 == 0:
                log.info(f"Step {step+1:04d} | Loss: {total_loss.item():.4f} | ddG: {-L_ddg.item():.4f}")

        # === 提取确定性摊销推断结果 ===
        generator_module.eval() 
        with torch.no_grad():
            _, final_delta_tensor = generator_module.mlp_adapter(generator_module.esm_logits_20)
            final_delta = final_delta_tensor.cpu().numpy()
        
        sig_muts = np.where(final_delta > threshold)
        all_candidates = []
        for pos, aa_idx in zip(sig_muts[0], sig_muts[1]):
            orig_aa = wt_seq[pos]
            mut_aa = STANDARD_AMINO_ACIDS[aa_idx]
            if orig_aa != mut_aa:
                real_pdb_id = seq_idx_to_pdb_id[pos]
                score = float(final_delta[pos, aa_idx])
                all_candidates.append({
                    "mutation": f"{orig_aa}{real_pdb_id}{mut_aa}", 
                    "score": round(score, 4)
                })
                
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_k = int(params.get('top_k', 50))
        return {
            "status": "success",
            "pdb_id": pdb_id.upper(),
            "chain_id": chain_id,
            "recommendations": all_candidates[:top_k] 
        }

    except Exception as e:
        log.error(f"❌ 设计任务失败: {str(e)}")
        return {"status": "error", "message": str(e)}