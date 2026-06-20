import math
import sys
import os
import torch
import pickle
import random
import json
import numpy as np
from sentence_transformers import SentenceTransformer
# from experiment.prompt.mmlu_prompt_set import Role_Connections
from typing import Dict
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

def get_kwargs(mode: str, N: int):
    initial_spatial_probability = 0.5
    initial_temporal_probability = 0.5
    fixed_spatial_masks = None
    fixed_temporal_masks = None
    node_kwargs = None
    fixed_spatial_masks_motif = None

    def generate_layered_graph(N, layer_num=2):
        adj = [[0] * N for _ in range(N)]
        base = N // layer_num
        rem = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base + (1 if i < rem else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            for j in range(N):
                if layers[j] == layers[i] + 1:
                    adj[i][j] = 1
        return adj

    def generate_mesh_graph(N):
        if N > 4 and int(math.sqrt(N))**2 == N:
            size = int(math.sqrt(N))
            adj = [[0] * N for _ in range(N)]
            for i in range(N):
                if (i + 1) % size != 0:
                    adj[i][i+1] = adj[i+1][i] = 1
                if i < N - size:
                    adj[i][i+size] = adj[i+size][i] = 1
            return adj
        return [[1 if i != j else 0 for i in range(N)] for j in range(N)]

    def generate_star_graph(N):
        adj = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj[0][i] = adj[i][0] = 1
        return adj

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        fixed_spatial_masks_motif = [[0]]
        node_kwargs = [{'role': 'Normal'}]
    elif mode in ('FullConnected', 'FakeFullConnected', 'FakeAGFull'):
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1] * N for _ in range(N)]
        fixed_spatial_masks_motif = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
    elif mode in ('Random', 'FakeRandom', 'FakeAGRandom'):
        fixed_spatial_masks = [[random.randint(0,1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0,1) for _ in range(N)] for _ in range(N)]
        fixed_spatial_masks_motif = [[random.randint(0,1) if i != j else 0 for i in range(N)] for j in range(N)]
    elif mode in ('Chain', 'FakeChain'):
        fixed_spatial_masks = [[1 if abs(i-j)==1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==j else 0 for i in range(N)] for j in range(N)]
        fixed_spatial_masks_motif = [[1 if abs(i-j)==1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
        fixed_spatial_masks_motif = generate_layered_graph(N)
    elif mode in ('Mesh', 'FakeMesh'):
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
        fixed_spatial_masks_motif = generate_mesh_graph(N)
    elif mode in ('Star', 'FakeStar'):
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
        fixed_spatial_masks_motif = generate_star_graph(N)
    elif 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': 'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': None} for i in range(N)]



    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "fixed_spatial_masks_motif": fixed_spatial_masks_motif,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs
    }


def save_graph_with_features(flow_graph, filepath, metadata):
    """
    Attach metadata to the graph and save it.
    """
    for key, value in metadata.items():
        setattr(flow_graph, key, value)
    torch.save(flow_graph, filepath)


def load_model(model_dir, role_embeddings_dict=None, ef=False):
    from experiment.args import Args
    if "IB" in model_dir:
        from experiment.model import ARGDesigner
    elif "OFA" in model_dir:
        from experiment.model_OFA import ARGDesigner
    else:
        from experiment.model import ARGDesigner
    model_file = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model file found at {model_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_file, map_location=device)
    # if not all(k in checkpoint for k in ('args', 'data_statistics', 'model_state_dict')):
    #     raise ValueError("Invalid checkpoint format. Missing required keys.")

    saved_args = checkpoint['args']
    # data_statistics = checkpoint['data_statistics']
    saved_args['data_dir'] = './ColdStartData_' + saved_args.get('dataset', '')
    args = Args()
    args.update_args_from_dict(saved_args)
    args.device = device

    model = ARGDesigner(args, role_embeddings_dict).to(device)
    if 'fixed_role_embeddings' in checkpoint['model_state_dict']:
        checkpoint['model_state_dict'].pop('fixed_role_embeddings')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model


def generate_graph(model, task_embedding, role_constraints_dict, question_id=None, max_nodes=5):
    """
    Generate a graph structure for the given task embedding.
    """
    with torch.no_grad():
        # generated = model.sample(num_samples=1, batch_size=1,
        #                          task_embedding=task_embedding,
        #                          question_id=question_id, vis=True)
        g = model.sample(task_embedding, max_nodes=max_nodes)                                 
    results = []
    g.graph['roles'] = []
    for n in g.nodes():
        role = g.nodes[n].get('role', 'Unknown')
        g.nodes[n]['constraint'] = role_constraints_dict.get(role, "")
        g.graph['roles'].append(role)
    results.append(g)
    return g


def convert_to_pyg_graph(nx_graph, task_text):
    """
    Convert a NetworkX graph into PyG Data format.
    """
    from torch_geometric.data import Data
    pyg = Data()
    num_nodes = nx_graph.number_of_nodes()
    features = []
    for i in range(num_nodes):
        features.append({
            'role': nx_graph.nodes[i].get('role', 'Unknown'),
            'constraint': nx_graph.nodes[i].get('constraint', '')
        })
    pyg.x = features
    edges = [[u, v] for u, v in nx_graph.edges()]
    pyg.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)
    pyg.task = task_text
    pyg.num_nodes = num_nodes
    return pyg


def precompute_role_embeddings(dsets, save_path="./prompt/precomputed_role_embeddings.pkl"):
    """
    Precompute role embeddings for different datasets.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    role_embeddings = {}

    if dsets == 'mmlu':
        from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'humaneval':
        from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'svamp':
        from experiment.svamp.svamp_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'aqua':
        from experiment.aqua.aqua_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'gsm8k':
        from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'multiarith':
        from experiment.multiarith.multiarith_prompt_set import ROLE_DESCRIPTION

    for role, description in ROLE_DESCRIPTION.items():
        full_embedding = model.encode(f"{role}: {description.strip()}")
        role_embeddings[role] = torch.tensor(full_embedding)

    with open(save_path, 'wb') as f:
        pickle.dump(role_embeddings, f)

    print(f"Precomputed {len(role_embeddings)} role embeddings, saved to {save_path}")
    return role_embeddings


# FIXME: generate_motif_embedding
def generate_motif_embedding(motif_name: str, motif_definition: Dict, 
                             role_descriptions: Dict, 
                             sentence_model: SentenceTransformer) -> torch.Tensor:
    """
    为单个 motif 生成嵌入向量
    
    策略：直接使用 motif 的 description 字段，通过 BERT 编码为嵌入向量
    motif description 已经包含了角色描述和逻辑关系，无需额外拼接
    
    Args:
        motif_name: motif 名称
        motif_definition: motif 定义，包含 'roles', 'connections', 'description'
        role_descriptions: 角色描述字典（保留参数以兼容，但不使用）
        sentence_model: Sentence Transformer 模型
    
    Returns:
        motif_embedding: (384,) 维向量
    """
    # 直接获取 motif 的描述
    # 该描述已经包含了：
    # 1. 各角色的能力说明
    # 2. 角色之间的逻辑关系
    # 3. 协作模式和工作流程
    # motif_description = motif_definition.get('description', '')
    
    # 如果没有描述，回退到使用 motif 名称
    # if not motif_description:
    #     print(f"Warning: Motif '{motif_name}' has no description, using name as fallback")
    #     motif_description = motif_name.replace('_', ' ')
    
    # 使用 Sentence Transformer (BERT-based) 将描述编码为 384 维向量
    embedding = sentence_model.encode(motif_definition, convert_to_tensor=True)
    
    return embedding

# FIXME: precompute_motif_embeddings
def precompute_motif_embeddings(dataset: str, 
                                embeddings_path: str,
                                force_recompute: bool = False) -> Dict[str, torch.Tensor]:
    """
    预计算所有 motif 的嵌入向量并保存
    
    Args:
        dataset: 数据集名称 ('gsm8k', 'mmlu', 'humaneval')
        embeddings_path: 保存路径
        force_recompute: 是否强制重新计算
    
    Returns:
        motif_embeddings: {motif_name: embedding_tensor}
    """
    # 如果已存在且不强制重算，直接加载
    if os.path.exists(embeddings_path) and not force_recompute:
        print(f"Loading precomputed motif embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing motif embeddings for dataset: {dataset}")
    
    # motif 不使用
    if dataset == 'gsm8k':
        from experiment.prompt.gsm8k_prompt_set import ROLE_DESCRIPTION,ROLE_DESCRIPTION_MOTIF
        role_descriptions = ROLE_DESCRIPTION
    elif dataset == 'mmlu':
        from experiment.prompt.mmlu_prompt_set import ROLE_DESCRIPTION_MOTIF,ROLE_DESCRIPTION_MOTIF
        role_descriptions = ROLE_DESCRIPTION_MOTIF
    elif dataset == 'humaneval':
        from experiment.prompt.humaneval_prompt_set import ROLE_DESCRIPTION,ROLE_DESCRIPTION_MOTIF
        role_descriptions = ROLE_DESCRIPTION
    elif dataset == 'aqua':
        from experiment.prompt.aqua_prompt_set import ROLE_DESCRIPTION,ROLE_DESCRIPTION_MOTIF
        role_descriptions = ROLE_DESCRIPTION
    elif dataset == 'svamp' or dataset == 'multiarith':
        from experiment.prompt.gsm8k_prompt_set import ROLE_DESCRIPTION,ROLE_DESCRIPTION_MOTIF
        role_descriptions = ROLE_DESCRIPTION
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # 初始化 Sentence Transformer
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 为每个 motif 生成嵌入
    motif_embeddings = {}
    for motif_name, motif_def in ROLE_DESCRIPTION_MOTIF.items():
        embedding = generate_motif_embedding(
            motif_name, motif_def, role_descriptions, sentence_model
        )
        motif_embeddings[motif_name] = embedding
        print(f"  Generated embedding for: {motif_name}")
    
    # 保存到文件
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    with open(embeddings_path, 'wb') as f:
        pickle.dump(motif_embeddings, f)
    
    print(f"Saved {len(motif_embeddings)} motif embeddings to {embeddings_path}")
    return motif_embeddings

class Accuracy:
    """
    Simple accuracy tracker.
    """
    def __init__(self):
        self._num_correct = 0
        self._num_total = 0

    def update(self, predicted: str, target: str) -> bool:
        is_correct = predicted == target
        self._num_correct += int(is_correct)
        self._num_total += 1
        return is_correct

    def get(self) -> float:
        return self._num_correct / self._num_total if self._num_total > 0 else 0.0

    def print(self):
        acc = self.get() * 100
        print(f"Accuracy: {acc:.1f}% ({self._num_correct}/{self._num_total})")

def get_attributes_len(
    len_node_map, len_edge_map, max_prev_node=None, max_head_and_tail=None
):
    """
    Returns (len_node_vec, len_edge_vec, feature_len)
    len_node_vec : Length of vector to represent a node attribute
    len_edge_vec : Length of vector to represent an edge attribute
    num_nodes_to_consider: Number of previous nodes to consider for edges for a given node
    """

    # Last two bits for START node and END node token
    len_node_vec = len_node_map
    # Last three bits in order are NO edge, START egde, END edge token
    len_edge_vec = len_edge_map + 3

    if max_prev_node is not None:
        num_nodes_to_consider = max_prev_node
    elif max_head_and_tail is not None:
        num_nodes_to_consider = max_head_and_tail[0] + max_head_and_tail[1]

    return len_node_vec, len_edge_vec, num_nodes_to_consider


def expand_motif_to_nodes(motif_configs: list, dataset: str, Role_Connections: dict):
    """
    将 motif 列表展开为节点列表（用于创建 Graph 对象）
    
    Args:
        motif_configs: motif 配置列表
        dataset: 数据集名称
        
    Returns:
        Tuple: (节点角色列表, motif_info字典)
        
    Example:
        >>> configs = [
        ...     {'motif_name': 'Analyst_Solver_Chain', ...},
        ...     {'motif_name': 'Single_Inspector', ...}
        ... ]
        >>> node_roles, motif_info = expand_motif_to_nodes(configs, 'gsm8k')
        >>> # node_roles = ['Mathematical Analyst', 'Math Solver', 'Inspector']
        >>> # motif_info = {0: {'motif_id': 0, 'node_idx_in_motif': 0}, ...}
    """
    all_node_roles = []
    motif_info = {}  # {global_node_id: {'motif_id': x, 'node_idx_in_motif': y}}
    motif_boundaries = []  # [(start_idx, end_idx), ...]
    
    global_node_counter = 0
    
    for motif_id, motif_config in enumerate(motif_configs):
        motif_start = global_node_counter
        roles = motif_config['role']
        
        # roles='Expert_Single'
        # 添加这个 motif 的所有角色节点
        Roles = Role_Connections[roles].get('role')

        for local_idx, role in enumerate(Roles):
            all_node_roles.append(role)
            motif_info[global_node_counter] = {
                'motif_id': motif_id,
                'motif_name': motif_config['role'],
                'node_idx_in_motif': local_idx,
                'role': role
            }
            global_node_counter += 1
        
        motif_end = global_node_counter - 1
        motif_boundaries.append((motif_start, motif_end))
    
    return all_node_roles, motif_info, motif_boundaries


def create_graph_from_motif(kwargs, dataset):
    if dataset == 'mmlu':
        from experiment.prompt.mmlu_prompt_set import Role_Connections
    elif dataset == 'gsm8k':
        from experiment.prompt.gsm8k_prompt_set import Role_Connections
    elif dataset == 'humaneval':
        from experiment.prompt.humaneval_prompt_set import Role_Connections
    elif dataset == 'aqua':
        from experiment.prompt.AQuA_prompt_set import Role_Connections
    node_roles, motif_info, motif_boundaries = expand_motif_to_nodes(kwargs['motif_kwargs'], dataset, Role_Connections)
    # 构建 motif 层面的 spatial masks（motif 之间的连接）
    motif_spatial_masks = kwargs['fixed_spatial_masks_motif']
    
    # 扩展为节点层面的 spatial masks（包含 motif 内部连接）
    total_nodes = len(node_roles)
    node_spatial_masks = [[0] * total_nodes for _ in range(total_nodes)]
    
    # 步骤1: 添加 motif 内部的连接
    num_motifs = len(motif_boundaries)
    for motif_config in kwargs['motif_kwargs']:
        internal_conns = Role_Connections[motif_config['role']].get('connections')
        motif_id = kwargs['motif_kwargs'].index(motif_config)
        motif_start, motif_end = motif_boundaries[motif_id]
        
        for src_local, dst_local in internal_conns:
            src_global = motif_start + src_local
            dst_global = motif_start + dst_local
            node_spatial_masks[src_global][dst_global] = 1
    
    # 步骤2: 添加 motif 之间的连接
    # motif 间的边连接各 motif 的"出口节点"和"入口节点"
    for src_motif in range(num_motifs):
        for dst_motif in range(num_motifs):
            if motif_spatial_masks[src_motif][dst_motif] == 1:
                # src_motif 的最后一个节点 → dst_motif 的第一个节点
                src_node = motif_boundaries[src_motif][1]  # 最后一个节点
                dst_node = motif_boundaries[dst_motif][0]  # 第一个节点
                node_spatial_masks[src_node][dst_node] = 1
    
    kwargs['fixed_spatial_masks'] = node_spatial_masks
    if dataset == 'mmlu':
        agent_names = ['AnalyzeAgent'] * total_nodes
    elif dataset == 'gsm8k':
        agent_names = ['MathSolver'] * total_nodes
    elif dataset == 'humaneval':
        agent_names = ['CodeWriting'] * total_nodes
    elif dataset == 'aqua':
        agent_names = ['MathSolver'] * total_nodes
    elif dataset == 'svamp' or dataset == 'multiarith':
        agent_names = ['MathSolver'] * total_nodes
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    # 构建节点配置
    kwargs['node_kwargs'] = [{'role': role} for role in node_roles]

    return kwargs, agent_names


def motif_to_graph(generated_graph_motif, dataset):
    """
    将 motif 级别的图转换为节点级别的 NetworkX 图
    
    Args:
        generated_graph_motif: NetworkX 图对象，包含 motif 级别的信息
            - graph['roles']: motif 名称列表，如 ['Expert_Critic_Chain', 'Mathematician_Single']
            - graph['mode']: 图的模式
            - graph['nodes']: 节点信息（如果有）
            - edges: motif 之间的边
            - 节点属性可能包含 'role' (motif 名称)
    
    Returns:
        nx.DiGraph: 节点级别的 NetworkX 有向图
    """
    import networkx as nx
    if dataset == 'mmlu':
        from experiment.prompt.mmlu_prompt_set import ROLE_DESCRIPTION, Role_Connections
    elif dataset == 'gsm8k':
        from experiment.prompt.gsm8k_prompt_set import ROLE_DESCRIPTION, Role_Connections   
    elif dataset == 'humaneval':
        from experiment.prompt.humaneval_prompt_set import ROLE_DESCRIPTION, Role_Connections
    elif dataset == 'aqua':
        from experiment.prompt.AQuA_prompt_set import ROLE_DESCRIPTION, Role_Connections
    
    # 获取 motif 序列
    motif_seq = generated_graph_motif.graph.get('roles', [])
    if not motif_seq:
        # 如果没有 roles，尝试从节点获取（按节点 ID 排序）
        sorted_nodes = sorted(generated_graph_motif.nodes())
        motif_seq = [generated_graph_motif.nodes[i].get('role', 'Unknown') 
                     for i in sorted_nodes]
    
    # 创建 motif_id 到原始节点 ID 的映射（如果节点 ID 不是连续的 0,1,2...）
    motif_id_to_node_id = {}
    if motif_seq:
        sorted_original_nodes = sorted(generated_graph_motif.nodes())
        for idx, node_id in enumerate(sorted_original_nodes):
            if idx < len(motif_seq):
                motif_id_to_node_id[idx] = node_id
    
    # 创建节点级别的图
    node_graph = nx.DiGraph()
    
    # 展开 motif 为节点
    all_node_roles = []
    motif_info = {}
    motif_boundaries = []
    global_node_counter = 0
    
    for motif_id, motif_name in enumerate(motif_seq):
        motif_start = global_node_counter
        
        # 从 Role_Connections 获取该 motif 包含的角色和内部连接
        if motif_name not in Role_Connections:
            # 如果 motif 名称不在 Role_Connections 中，使用默认处理
            print(f"Warning: Motif '{motif_name}' not found in Role_Connections, using as single node")
            roles = [motif_name]
            internal_connections = []
        else:
            motif_config = Role_Connections[motif_name]
            roles = motif_config.get('role', [motif_name])
            internal_connections = motif_config.get('connections', [])
        
        # 添加该 motif 的所有角色节点
        for local_idx, role in enumerate(roles):
            node_id = global_node_counter
            node_graph.add_node(node_id, role=role, constraint=ROLE_DESCRIPTION[role])
            all_node_roles.append(role)
            motif_info[global_node_counter] = {
                'motif_id': motif_id,
                'motif_name': motif_name,
                'node_idx_in_motif': local_idx,
                'role': role
            }
            global_node_counter += 1
        
        # 添加 motif 内部的连接
        for src_local, dst_local in internal_connections:
            src_global = motif_start + src_local
            dst_global = motif_start + dst_local
            if src_global < global_node_counter and dst_global < global_node_counter:
                node_graph.add_edge(src_global, dst_global, label=1)
        
        motif_end = global_node_counter - 1
        motif_boundaries.append((motif_start, motif_end))
    
    # 获取 motif 之间的边（从原始 motif 图）
    motif_edges = list(generated_graph_motif.edges())
    
    # 将 motif 之间的边转换为节点之间的边
    # 策略：motif A 的最后一个节点 -> motif B 的第一个节点
    for src_node_id, dst_node_id in motif_edges:
        # 将原始节点 ID 映射到 motif_id
        src_motif_id = None
        dst_motif_id = None
        
        # 查找对应的 motif_id
        for mid, nid in motif_id_to_node_id.items():
            if nid == src_node_id:
                src_motif_id = mid
            if nid == dst_node_id:
                dst_motif_id = mid
        
        # 如果找不到映射，尝试直接使用节点 ID（假设节点 ID 就是 motif_id）
        if src_motif_id is None:
            src_motif_id = src_node_id
        if dst_motif_id is None:
            dst_motif_id = dst_node_id
        
        if (src_motif_id < len(motif_boundaries) and dst_motif_id < len(motif_boundaries) and
            src_motif_id >= 0 and dst_motif_id >= 0):
            src_start, src_end = motif_boundaries[src_motif_id]
            dst_start, dst_end = motif_boundaries[dst_motif_id]
            
            # src_motif 的最后一个节点 -> dst_motif 的第一个节点
            src_node = src_end
            dst_node = dst_start
            
            if src_node < global_node_counter and dst_node < global_node_counter:
                node_graph.add_edge(src_node, dst_node, label=1)
    
    # 复制图的元数据
    node_graph.graph['mode'] = generated_graph_motif.graph.get('mode', 'Generated')
    node_graph.graph['agent_nums'] = len(all_node_roles)
    node_graph.graph['roles'] = all_node_roles
    node_graph.graph['motif_info'] = motif_info
    node_graph.graph['motif_boundaries'] = motif_boundaries
    
    # 如果原始图有 nodes 信息，也复制过来
    if 'nodes' in generated_graph_motif.graph:
        node_graph.graph['nodes'] = generated_graph_motif.graph['nodes']
    
    return node_graph