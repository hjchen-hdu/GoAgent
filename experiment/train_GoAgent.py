
import torch
import os
from torch.utils.data import DataLoader
import networkx as nx


def collate_for_ofa(batch_graphs, device, role_to_id, pad_token_id):
    if not batch_graphs:
        return {}
    max_nodes = 0
    for g in batch_graphs:
        max_nodes = max(max_nodes, len(g.graph['final_roles']))

    batch_size = len(batch_graphs)
    embedding_dim = batch_graphs[0].graph['task_embedding'].shape[0]
    task_embeddings = torch.zeros(batch_size, embedding_dim, device=device)
    adj_gt = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    node_roles = torch.full((batch_size, max_nodes), pad_token_id, dtype=torch.long, device=device)
    graph_sizes = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, g in enumerate(batch_graphs):
        roles = g.graph['final_roles']
        num_nodes = len(roles)
        
        task_embeddings[i] = g.graph['task_embedding'].to(device)
        graph_sizes[i] = num_nodes
        
        role_ids = torch.tensor([role_to_id[r] for r in roles], dtype=torch.long, device=device)
        node_roles[i, :num_nodes] = role_ids
        
        num_real_nodes = num_nodes - 1
        if num_real_nodes > 0:
            nodes = sorted(list(g.nodes()))
            original_adj = torch.tensor(nx.to_numpy_array(g, nodelist=nodes), dtype=torch.float32, device=device)
            adj_gt[i, :num_real_nodes, :num_real_nodes] = original_adj
            
    return {
        'task_embedding': task_embeddings,
        'adj_gt': adj_gt,
        'node_roles': node_roles,
        'graph_sizes': graph_sizes
    }


def train(args, model, dataset_train, dataset_validate=None):
    print('Training model...')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    if dataset_validate is None:
        dataset_validate = dataset_train

    collate_fn_with_args = lambda batch: collate_for_ofa(batch, args.device, model.role_to_id, model.PAD_TOKEN_ID)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_args)
    dataloader_validate = DataLoader(dataset_validate, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_args)

    best_validate_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        batch_count = 0
        loss_sum = 0
        role_accuracy_sum = 0
        total_loss_node_sum = 0
        total_loss_edge_sum = 0
        total_kl_node_sum = 0
        total_kl_edge_sum = 0
        for graphs in dataloader_train:
            optimizer.zero_grad()

            loss, total_loss_node, total_loss_edge, total_kl_node, total_kl_edge, batch_role_accuracy = model(graphs, epoch)

            # 检查损失是否有效（NaN/Inf）
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected (NaN/Inf), skipping batch")
                continue

            role_accuracy_sum += batch_role_accuracy

            loss.backward()
            
            # 检查梯度是否有效
            if args.clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Invalid gradient norm detected, skipping batch")
                    optimizer.zero_grad()  # 清除无效梯度
                    continue

            optimizer.step()
            
            loss_sum += loss.item()
            total_loss_node_sum += total_loss_node.item()
            # 总是累加边损失，即使为0，以保持统计一致性
            total_loss_edge_sum += total_loss_edge.item()
            total_kl_node_sum += total_kl_node.item()
            total_kl_edge_sum += total_kl_edge.item()
            batch_count += 1

        epoch_loss = loss_sum / batch_count if batch_count > 0 else float('inf')
        avg_loss_node = total_loss_node_sum / batch_count if batch_count > 0 else 0.0
        avg_loss_edge = total_loss_edge_sum / batch_count if batch_count > 0 else 0.0
        avg_kl_node = total_kl_node_sum / batch_count if batch_count > 0 else 0.0
        avg_kl_edge = total_kl_edge_sum / batch_count if batch_count > 0 else 0.0
        avg_role_accuracy = role_accuracy_sum / batch_count if batch_count > 0 else 0.0
        scheduler.step(epoch_loss)
        
        print(
            f'Epoch {epoch + 1}/{args.epochs}, Average Training Loss: {epoch_loss:.4f}, Node Loss: {avg_loss_node:.4f}, Edge Loss: {avg_loss_edge:.4f}, KL Node Loss: {avg_kl_node:.4f}, KL Edge Loss: {avg_kl_edge:.4f}, Role Prediction Accuracy: {avg_role_accuracy:.2f}%')
        # print("lr:", optimizer.param_groups[0]['lr'])
        if dataloader_validate and (epoch + 1) % args.epochs_validate == 0:
            validate_loss, avg_loss_node, avg_loss_edge, avg_kl_node, avg_kl_edge, val_role_accuracy = validate(args, model, dataloader_validate)
            print(
                f'Epoch {epoch + 1}/{args.epochs}, Validation Loss: {validate_loss:.4f}, Node Loss: {avg_loss_node:.4f}, Edge Loss: {avg_loss_edge:.4f}, KL Node Loss: {avg_kl_node:.4f}, KL Edge Loss: {avg_kl_edge:.4f}, Validation Role Accuracy: {val_role_accuracy:.2f}%')

            if validate_loss < best_validate_loss:
                best_validate_loss = validate_loss
                best_epoch = epoch + 1
                if args.save_model:
                    best_model_path = os.path.join(args.experiment_path, args.model_name)

                    save_content = {
                        'model_state_dict': model.state_dict(),
                        # 'data_statistics': model.data_statistics,
                        'args': args.__dict__
                    }
                    torch.save(save_content, best_model_path)
                    print(
                        f"Saved best model and statistics to {best_model_path}, Epoch {epoch + 1}, Validation loss: {validate_loss:.4f}")

    print(f'Training completed. Best model at Epoch {best_epoch}, Validation loss: {best_validate_loss:.4f}')


def validate(args, model, dataloader_validate):
    model.eval()
    loss_sum = 0
    batch_count = 0
    role_accuracy_sum = 0.0
    total_loss_node_sum = 0.0
    total_loss_edge_sum = 0.0
    total_kl_node_sum = 0.0
    total_kl_edge_sum = 0.0
    with torch.no_grad():
        for batch_idx, batch_graphs in enumerate(dataloader_validate):
            # 在测试/验证时，一般还是要传递epoch参数，确保KL warmup等机制一致；此时应设置为最大warmup之后的数，如args.epochs或warmup_delay+warmup_epochs
            loss, total_loss_node, total_loss_edge, total_kl_node, total_kl_edge, batch_role_accuracy = model(batch_graphs, epoch=args.epochs)
            
            # 检查损失是否有效（NaN/Inf）
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid validation loss detected (NaN/Inf), skipping batch")
                continue
            
            role_accuracy_sum += batch_role_accuracy
            batch_count += 1
            loss_sum += loss.item()
            total_loss_node_sum += total_loss_node.item()
            total_loss_edge_sum += total_loss_edge.item()
            total_kl_node_sum += total_kl_node.item()
            total_kl_edge_sum += total_kl_edge.item()
    avg_validate_loss = loss_sum / max(batch_count, 1)
    avg_loss_node = total_loss_node_sum / max(batch_count, 1)
    avg_loss_edge = total_loss_edge_sum / max(batch_count, 1)
    avg_kl_node = total_kl_node_sum / max(batch_count, 1)
    avg_kl_edge = total_kl_edge_sum / max(batch_count, 1)
    avg_role_accuracy = role_accuracy_sum / max(batch_count, 1)

    return avg_validate_loss, avg_loss_node, avg_loss_edge, avg_kl_node, avg_kl_edge, avg_role_accuracy
