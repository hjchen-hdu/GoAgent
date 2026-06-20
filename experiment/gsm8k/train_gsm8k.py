import os
import json
import torch
import numpy as np
import argparse
import random
import networkx as nx
from tqdm import tqdm
import asyncio
import copy
import sys
import shutil
import time
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.args import Args
from experiment.model import GoAgent
from experiment.train_GoAgent import train
from experiment.utils import save_graph_with_features, get_kwargs, load_model, generate_graph, convert_to_pyg_graph
from sentence_transformers import SentenceTransformer
from experiment import process_dataset as gdata
from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from experiment.prompt.gsm8k_prompt_set import ROLE_DESCRIPTION, ROLE_DESCRIPTION_MOTIF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
COLD_START_DIR = os.path.join(PROJECT_ROOT, "experiment", "gsm8k", "ColdStartData_gsm8k")
TASK_SPLIT_FILE = "./file/task_split_gsm8k.json"    


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Full pipeline for GSM8K: pretrain, generate data, finetune")
    parser.add_argument('--pretrain', action='store_true', help="Run pretraining if set")
    parser.add_argument('--load_from_dir', type=str, default=None, help="Directory to load pretrained model from")

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of pretraining epochs")
    parser.add_argument('--pretrain_lr', type=float, default=5e-4, help="Learning rate for pretraining")

    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument('--domain', type=str, default="gsm8k", help="Task domain")
    parser.add_argument('--decision_method', type=str, default="FinalRefer", help="Decision method")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'], help='List of agent names')
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds")

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and data gen")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--model_output_dir', type=str, default='output/gsm8k_finetuned_model_IB',
                        help="Root directory for model outputs")

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


def run_pretraining(args):
    print("\n" + "=" * 20 + " Stage 1: Pretraining GSM8K Model " + "=" * 20)
    pre = Args().update_args()
    for k, v in vars(args).items():
        setattr(pre, k, v)
    pre.dataset = 'gsm8k'
    pre.data_dir = COLD_START_DIR
    pre.experiment_path = args.model_output_dir
    pre.pretrain = args.pretrain
    pre.epochs = args.pretrain_epochs
    pre.lr = args.pretrain_lr
    pre.batch_size = args.batch_size
    pre.seed = args.seed
    pre.model_name = 'best_model.pth'

    ds, role_embeddings_dict = gdata.load_graph_dataset(pre)

    correct = [g for g in ds if g.graph.get('is_correct')]
    incorrect = [g for g in ds if not g.graph.get('is_correct')]
    random.shuffle(correct)
    random.shuffle(incorrect)

    correct = [g for g in ds if g.graph.get('is_correct')]
    incorrect = [g for g in ds if not g.graph.get('is_correct')]
    random.shuffle(correct)
    random.shuffle(incorrect)

    ratio = 0.9
    train_graphs = correct[:int(len(correct) * ratio)] + incorrect[:int(len(incorrect) * ratio)]
    val_graphs = correct[int(len(correct) * ratio):] + incorrect[int(len(incorrect) * ratio):]
    random.shuffle(train_graphs)
    random.shuffle(val_graphs)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    with open(os.path.join(pre.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(pre.__dict__, f, indent=2)

    model = ARGDesigner(pre, role_embeddings_dict).to(pre.device)
    train(pre, model, train_graphs, val_graphs)

    print("==================== Complete ====================")
    print(f"Best model at: {os.path.join(pre.experiment_path)}")
    return pre.experiment_path, os.path.join(pre.experiment_path, 'configuration.txt')


def find_latest_model_dir(bd='output', pf='gsm8k_finetuned_model'):
    if not os.path.exists(bd): return None
    cds = [d for d in os.listdir(bd) if os.path.isdir(os.path.join(bd, d)) and d.startswith(pf)]
    return os.path.join(bd, sorted(cds)[-1]) if cds else None


async def main():
    args = parse_cli_args()
    args.pretrain = True
    setup_environment(args)
    loop = asyncio.get_running_loop()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.model_output_dir = f"{args.model_output_dir}_{timestamp}"
    os.makedirs(args.model_output_dir, exist_ok=True)
    print(f"Starting new GSM8K run, outputs to: {args.model_output_dir}")
    pretrained_model_path, config_file_path = await loop.run_in_executor(None, run_pretraining, args)


    # print(f"\nGSM8K pipeline complete! Final efficient model at: {os.path.join(args.model_output_dir)}")


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
