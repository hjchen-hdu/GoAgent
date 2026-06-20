import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from experiment.utils import precompute_motif_embeddings, get_attributes_len
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data._utils.collate import default_collate as collate
import torch.nn.init as init

import networkx as nx


class ConditionalVIBLayer(nn.Module):
    def __init__(self, dim, clamp_logvar=(-6.0, 4.0)): 
        super().__init__()
        self.q_mu = nn.Linear(dim, dim)
        self.q_logvar = nn.Linear(dim, dim)
        self.p_mu = nn.Linear(dim, dim)
        self.p_logvar = nn.Linear(dim, dim)
        self.clamp_logvar = clamp_logvar
        self.min_var = 1e-6 

    def forward(self, h, cond):
        mu_q = self.q_mu(h)
        logvar_q = self.q_logvar(h).clamp(*self.clamp_logvar)
        mu_p = self.p_mu(cond)
        logvar_p = self.p_logvar(cond).clamp(*self.clamp_logvar)

        if self.training:
            std_q = torch.exp(0.5 * logvar_q)
            z = mu_q + torch.randn_like(std_q) * std_q
        else:
            z = mu_q

        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p).clamp(min=self.min_var)  
        
        kl_per_dim = 0.5 * (
            logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
        )
        kl = kl_per_dim.mean(dim=-1) 
        return z, kl



class MLP_Basic(nn.Module):
    """Basic MLP implementation"""

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Basic, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)



class GoAgent(nn.Module):
    def __init__(self, args, role_embeddings_dict):
        super(GoAgent, self).__init__()
        self.args = args
        self.hidden_dim = 256
        self.min_nodes = 1
        self.embedding_dim = 384

        self.role_names = list(role_embeddings_dict.keys())
        base_role_embeddings = torch.stack(list(role_embeddings_dict.values()), dim=0)

        start_embedding = torch.zeros(1, self.embedding_dim)
        self.end_embedding = nn.Parameter(torch.randn(self.embedding_dim))
        pad_embedding = torch.zeros(1, self.embedding_dim)
        fixed_embeddings = torch.cat([base_role_embeddings, start_embedding, pad_embedding], dim=0)
        self.register_buffer('fixed_role_embeddings', fixed_embeddings)

        self.id_to_role = {i: name for i, name in enumerate(self.role_names)}
        self.role_to_id = {name: i for i, name in enumerate(self.role_names)}

        self.START_TOKEN_ID = len(self.role_names)
        self.END_TOKEN_ID = len(self.role_names) + 1
        self.PAD_TOKEN_ID = len(self.role_names) + 2

        self.id_to_role[self.START_TOKEN_ID] = 'START_TOKEN'
        self.id_to_role[self.END_TOKEN_ID] = 'END_TOKEN'
        self.id_to_role[self.PAD_TOKEN_ID] = 'PAD_TOKEN'
        self.role_to_id['START_TOKEN'] = self.START_TOKEN_ID
        self.role_to_id['END_TOKEN'] = self.END_TOKEN_ID
        self.role_to_id['PAD_TOKEN'] = self.PAD_TOKEN_ID

        self.task_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )

        self.edge_net = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        self.prev_nodes_aggregator = nn.GRU(
            self.embedding_dim,
            self.embedding_dim,
            batch_first=True
        )
        self.role_proj = MLP_Basic(self.embedding_dim, self.hidden_dim, self.embedding_dim)

        self.node_gru = nn.GRU(
            self.embedding_dim,
            self.embedding_dim,
            batch_first=True
        )
        self.edge_gru = nn.GRU(
            self.embedding_dim,
            self.embedding_dim,
            batch_first=True
        )
        self.node_project = MLP_Basic(
            self.embedding_dim,
            self.hidden_dim,
            self.embedding_dim
        )
        self.edge_project = MLP_Basic(
            self.embedding_dim*3,
            self.hidden_dim,
            self.embedding_dim
        )

        self.output_node = MLP_Basic(
            self.embedding_dim,
            self.hidden_dim,
            self.embedding_dim
        )
        self.output_edge = MLP_Basic(
            self.embedding_dim,
            self.hidden_dim,
            self.embedding_dim
        )

        self.max_steps = getattr(args, "max_nodes", 7) 
        self.step_embedding = nn.Embedding(self.max_steps, self.embedding_dim)
        nn.init.normal_(self.step_embedding.weight, mean=0.0, std=0.02)

        if getattr(args, "use_vib", False):
            self.beta_node = args.beta_node
            self.beta_edge = args.beta_edge
            self.vib_node = ConditionalVIBLayer(self.embedding_dim)
            self.vib_edge = ConditionalVIBLayer(self.embedding_dim) 
        else:
            self.beta_node = 0
            self.beta_edge = 0

    @property
    def role_embeddings(self):
        base_and_start = self.fixed_role_embeddings[:-1]  
        pad = self.fixed_role_embeddings[-1:] 
        return torch.cat([base_and_start, self.end_embedding.unsqueeze(0), pad], dim=0)

    def forward(self, batch, epoch=0):
        batch_size = batch['task_embedding'].shape[0]
        device = batch['task_embedding'].device

        task_embedding = batch['task_embedding']

        z = self.task_encoder(task_embedding)

        adj_gt = batch['adj_gt']       
        node_roles_gt = batch['node_roles']
        graph_sizes = batch['graph_sizes']
        max_nodes = adj_gt.shape[1]

        total_loss_graph = torch.tensor(0.0, device=device)
        total_loss_node = torch.tensor(0.0, device=device)
        total_loss_edge = torch.tensor(0.0, device=device)
        total_correct_predictions = 0
        total_node_predictions = 0
        total_edge_predictions = 0
        total_kl_node = torch.tensor(0.0, device=device)
        total_kl_edge = torch.tensor(0.0, device=device)

        
        t_range = torch.arange(0, max_nodes, device=device).unsqueeze(0)
        active_mask = (t_range < graph_sizes.unsqueeze(1)).float()
        h_encoded = torch.zeros(batch_size, max_nodes, self.embedding_dim, device=device)
        for t in range(max_nodes):
            num_active_graphs = int(active_mask[:, t].sum())
            if num_active_graphs == 0:
                continue

            active_indices = torch.where(active_mask[:, t] == 1)[0]
            
            history_nodes_ids = node_roles_gt[active_indices, :t]
            
            history_nodes_embeddings = self.role_embeddings[history_nodes_ids] 

            z_active = z[active_indices]
            
            if t > 0:
                _, h_his = self.prev_nodes_aggregator(history_nodes_embeddings)
                h_his = h_his[-1]
                gate = torch.sigmoid(torch.sum(h_his * z_active, dim=-1, keepdim=True) / self.embedding_dim)
                combined = (1 - gate) * h_his + gate * z_active
            else:
                combined = torch.zeros_like(z_active)
            
            step_t = min(t, self.max_steps - 1)
            step_emb = self.step_embedding(torch.tensor(step_t, device=device))
            combined = combined + step_emb 
            
            h_encoded[active_indices, t] = combined

            node_level_input = self.node_project(h_encoded[active_indices, :t+1])
            node_level_output, h_node = self.node_gru(node_level_input)
            x_pred_node = self.output_node(torch.mean(node_level_output, dim=1))
            
            if self.beta_node > 0:
                q_t, kl_node = self.vib_node(x_pred_node, z_active)
                total_kl_node += kl_node.sum()
            else:
                q_t = x_pred_node  

            node_logits = q_t @ self.role_embeddings.t() 
            gt_node_ids = node_roles_gt[active_indices, t]
            
            loss_node = F.cross_entropy(node_logits, gt_node_ids, reduction='none')
            total_loss_node += loss_node.sum()

            pred_node_ids = torch.argmax(node_logits, dim=1)
            total_correct_predictions += (pred_node_ids == gt_node_ids).sum().item()
            total_node_predictions += num_active_graphs

            gt_new_node_embeddings = self.role_embeddings[gt_node_ids]
            valid_edge_pred_mask = (gt_node_ids != self.END_TOKEN_ID) & (gt_node_ids != self.PAD_TOKEN_ID)
            
            if t > 0 and valid_edge_pred_mask.any():
                edge_pred_indices = torch.where(valid_edge_pred_mask)[0]
                
                edge_net_inputs = []
                gt_edges_list = []
                
                for i in range(num_active_graphs):
                    if valid_edge_pred_mask[i]:
                        graph_h = h_encoded[active_indices[i], :t]
                        num_existing = graph_h.shape[0]

                        if num_existing > 0:
                            new_node_emb = gt_new_node_embeddings[i].unsqueeze(0).repeat(num_existing, 1)
                            task_emb_rep = z[active_indices[i]].unsqueeze(0).repeat(num_existing, 1)

                            edge_net_input = torch.cat([graph_h, new_node_emb, task_emb_rep], dim=1)
                            edge_level_input = self.edge_project(edge_net_input)
                            edge_level_output, h_edge = self.edge_gru(edge_level_input)
                            x_pred_edge = self.output_edge(edge_level_output)

                            edge_net_inputs.append(x_pred_edge)

                            
                            gt_edges_list.append(adj_gt[active_indices[i], :num_existing, t])

                if len(edge_net_inputs) > 0:
                    edge_net_input_batch = torch.cat(edge_net_inputs, dim=0)
                    gt_edges_batch = torch.cat(gt_edges_list, dim=0) 
                    
                    if self.beta_edge > 0:
                        valid_active_indices = active_indices[edge_pred_indices]
                        task_cond = z[valid_active_indices].repeat_interleave(
                            torch.tensor([h_encoded[active_indices[i], :t].shape[0] 
                                         for i in range(num_active_graphs) if valid_edge_pred_mask[i]], 
                                        device=device), dim=0
                        )
                        edge_feat_compressed, kl_edge = self.vib_edge(edge_net_input_batch, task_cond)
                        total_kl_edge += kl_edge.sum()
                    else:
                        edge_feat_compressed = edge_net_input_batch

                    edge_logits = self.edge_net(edge_feat_compressed).squeeze(-1)
                    total_edge_predictions += edge_logits.shape[0]
                    loss_edge = F.binary_cross_entropy_with_logits(edge_logits, gt_edges_batch.float(), reduction='none')
                    total_loss_edge += loss_edge.sum()

        avg_kl_node = total_kl_node / total_node_predictions if total_node_predictions > 0 else torch.tensor(0.0, device=device)
        avg_kl_edge = total_kl_edge / total_node_predictions if total_node_predictions > 0 else torch.tensor(0.0, device=device)
        avg_node_loss = total_loss_node / total_node_predictions if total_node_predictions > 0 else torch.tensor(0.0, device=device)
        if total_edge_predictions > 0:
            avg_edge_loss = total_loss_edge / total_node_predictions
        else:
            avg_edge_loss = torch.tensor(0.0, device=device)
        
        warmup_delay = 5
        warmup_epochs = 10
        if epoch < warmup_delay:
            beta_node_effective = 0.0
            beta_edge_effective = 0.0
        else:
            progress = min(1.0, (epoch - warmup_delay + 1) / warmup_epochs)
            beta_node_effective = self.beta_node * progress
            beta_edge_effective = self.beta_edge * progress
        
        total_loss_graph = avg_node_loss + avg_edge_loss + beta_node_effective * avg_kl_node + beta_edge_effective * avg_kl_edge
        batch_accuracy = (total_correct_predictions / total_node_predictions) * 100 if total_node_predictions > 0 else 0
        return total_loss_graph, avg_node_loss, avg_edge_loss, avg_kl_node, avg_kl_edge, batch_accuracy

    def sample(self, task_query_embedding, max_nodes=5):
        self.eval() 
        device = next(self.parameters()).device
        min_num_nodes = self.min_nodes

        with torch.no_grad():
            task_query_embedding = task_query_embedding.to(device)
            z = self.task_encoder(task_query_embedding.unsqueeze(0))


            candidate_ids = list(range(len(self.role_to_id)-3)) + [self.END_TOKEN_ID]
            candidate_embeddings = self.role_embeddings[candidate_ids]
            end_token_candidate_idx = len(candidate_ids) - 1

            G = nx.DiGraph()
            h_existing = torch.empty(0, self.embedding_dim).to(device)

            t = 0
            h_encoded = torch.empty(0, self.embedding_dim).to(device)
            while t < max_nodes:
                if t > 0:
                    _, h_his = self.prev_nodes_aggregator(h_existing)
                    gate = torch.sigmoid(torch.sum(h_his * z, dim=-1, keepdim=True) / self.embedding_dim)
                    combined = (1 - gate) * h_his + gate * z
                else:
                    combined = torch.zeros_like(z)
                
                step_t = min(t, self.max_steps - 1)
                step_emb = self.step_embedding(torch.tensor(step_t, device=device))
                combined = combined + step_emb 
                    
                h_encoded = torch.cat([h_encoded, combined], dim=0)
                node_level_input = self.node_project(h_encoded)
                node_level_output, h_node = self.node_gru(node_level_input)
                x_pred_node = self.output_node(torch.mean(node_level_output, dim=0)).unsqueeze(0)
                
                if self.beta_node > 0:
                    q_t, _ = self.vib_node(x_pred_node, z)
                else:
                    q_t = x_pred_node 

                node_logits = q_t @ candidate_embeddings.t()
                node_probs = F.softmax(node_logits, dim=1)

                if t < min_num_nodes:
                    node_probs[0, end_token_candidate_idx] = 0

                if torch.sum(node_probs).item() < 1e-8:
                    node_probs = torch.ones_like(node_probs)
                    if t < min_num_nodes:
                        node_probs[0, end_token_candidate_idx] = 0
                
                if torch.sum(node_probs).item() > 0:
                    node_probs = node_probs / torch.sum(node_probs)
                else:
                    break

                sampled_candidate_idx = torch.multinomial(node_probs, 1).item()
                new_node_id = candidate_ids[sampled_candidate_idx]

                if new_node_id == self.END_TOKEN_ID:
                    break

                new_node_role = self.id_to_role[new_node_id]
                new_node_embedding = self.role_embeddings[new_node_id]

                G.add_node(t, role=new_node_role)
                h_existing = torch.cat([h_existing, new_node_embedding.unsqueeze(0)], dim=0)

                if G.number_of_nodes() > 1:
                    h_encoded_prev = h_encoded[:-1] 
                    num_existing = h_encoded_prev.shape[0]
                    edge_net_input = torch.cat([
                        h_encoded_prev, 
                        new_node_embedding.unsqueeze(0).repeat(num_existing, 1),  
                        z.repeat(num_existing, 1)  
                    ], dim=1)

                    edge_level_input = self.edge_project(edge_net_input)
                    edge_level_output, h_edge = self.edge_gru(edge_level_input)
                    x_pred_edge = self.output_edge(edge_level_output)
                    
                    if self.beta_edge > 0:
                        task_cond = z.repeat(num_existing, 1)
                        edge_feat_compressed, _ = self.vib_edge(x_pred_edge, task_cond)
                    else:
                        edge_feat_compressed = x_pred_edge
                    
                    edge_logits = self.edge_net(edge_feat_compressed).squeeze(-1)
                    edge_probs = torch.sigmoid(edge_logits)

                    edge_samples = torch.bernoulli(edge_probs)
                    edges_added = False

                    for i, should_add in enumerate(edge_samples):
                        if should_add.item() == 1:
                            G.add_edge(i, t)
                            edges_added = True

                    if not edges_added:
                        best_edge_idx = torch.argmax(edge_probs).item()
                        G.add_edge(best_edge_idx, t)

                t += 1

            return G
