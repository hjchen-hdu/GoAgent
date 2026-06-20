import os
import json
import torch
import numpy as np
import argparse
import random
import networkx as nx
from tqdm import tqdm
import asyncio
import math
import copy
import sys
import time
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.args import Args
from experiment.model import GoAgent
from experiment.train_GoAgent import train
from experiment.utils import load_model, generate_graph, convert_to_pyg_graph, Accuracy, get_kwargs, save_graph_with_features
from sentence_transformers import SentenceTransformer
from experiment import process_dataset as gdata

from mas_framework.graph.graph import Graph, TestGraph
from datasets.mmlu_dataset import MMLUDataset

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Full pipeline: pretrain, data generation, finetune")
    parser.add_argument('--pretrain', action='store_true', help="Run pretraining from scratch")
    parser.add_argument('--load_from_dir', type=str, default=None, help="Directory to load pretrained model")

    parser.add_argument('--pretrain_epochs', type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument('--pretrain_lr', type=float, default=8e-5, help="Learning rate for pretraining")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for pretraining")

    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument('--domain', type=str, default="mmlu", help="Task domain")
    parser.add_argument('--decision_method', type=str, default="FinalRefer", help="Decision method for final node")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'], help="Names of agents")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds")

    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training and generation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed") # 114514，3407，42
    parser.add_argument('--model_output_dir', type=str, default='output/efficiency_finetuned_model_IB',
                        help="Output directory for all model artifacts")

    parser.add_argument('--use_vib', action='store_true', default=True, help="Use VIB bottleneck")
    parser.add_argument('--beta_node', type=float, default=0.1, help="Beta for node bottleneck")
    parser.add_argument('--beta_edge', type=float, default=0.1, help="Beta for edge bottleneck")
    return parser.parse_args()


def setup_environment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def train_model(args):
    print("\n==================== Training ====================")
    pretrain_args = Args().update_args()
    for k, v in vars(args).items():
        setattr(pretrain_args, k, v)
    pretrain_args.data_dir = COLD_START_DIR
    pretrain_args.experiment_path = args.model_output_dir
    pretrain_args.pretrain = args.pretrain
    pretrain_args.epochs = args.pretrain_epochs
    pretrain_args.lr = args.pretrain_lr
    pretrain_args.weight_decay = args.weight_decay
    pretrain_args.batch_size = args.batch_size
    pretrain_args.seed = args.seed
    pretrain_args.model_name = 'best_model.pth'

    graph_dataset, role_embeddings_dict = gdata.load_graph_dataset(pretrain_args)

    correct_graphs = [g for g in graph_dataset if g.graph.get('is_correct')]
    incorrect_graphs = [g for g in graph_dataset if not g.graph.get('is_correct')]
    random.shuffle(correct_graphs)
    random.shuffle(incorrect_graphs)

    train_ratio = 0.9
    train_graphs = correct_graphs[:int(len(correct_graphs) * train_ratio)] + incorrect_graphs[:int(len(incorrect_graphs) * train_ratio)]
    val_graphs = correct_graphs[int(len(correct_graphs) * train_ratio):] + incorrect_graphs[int(len(incorrect_graphs) * train_ratio):]
    random.shuffle(train_graphs)
    random.shuffle(val_graphs)

    print(f"Train set: {len(train_graphs)}, Validation set: {len(val_graphs)}")

    with open(os.path.join(pretrain_args.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(pretrain_args.__dict__, f, indent=2)

    model = GoAgent(pretrain_args, role_embeddings_dict).to(pretrain_args.device)
    train(pretrain_args, model, train_graphs, val_graphs)

    print("==================== Complete ====================")
    return os.path.join(pretrain_args.experiment_path), os.path.join(pretrain_args.experiment_path, 'configuration.txt')



def find_latest_model_dir(base_dir='output'):
    prefix = 'efficiency_finetuned_model'
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]
    return os.path.join(base_dir, sorted(dirs)[-1]) if dirs else None


async def main(flag):
    args = parse_cli_args()
    args.pretrain = flag
    setup_environment(args)
    loop = asyncio.get_running_loop()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.model_output_dir = f"{args.model_output_dir}_{timestamp}"
    os.makedirs(args.model_output_dir, exist_ok=True)
    print(f"Starting new run, outputs in: {args.model_output_dir}")
    pretrained_path, config_path = await loop.run_in_executor(None, train_model, args)
    print("\nPipeline complete. Final model at:", pretrained_path)



if __name__ == '__main__':
    asyncio.run(main(True))
