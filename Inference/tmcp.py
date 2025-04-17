import torch
import numpy as np

class TemporalGCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(TemporalGCNLayer, self).__init__()
        self.W = torch.nn.Linear(in_features, out_features)
        self.time_embedding = torch.nn.Linear(1, out_features)
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(out_features, out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(out_features, out_features)
        )
    
    def forward(self, node_features, adjacency_matrix, timestamps):
        node_features = node_features
        adjacency_matrix = adjacency_matrix
        timestamps = timestamps
        
        temporal_features = self.time_embedding(timestamps.unsqueeze(-1))
        node_features = self.W(node_features) + temporal_features
        node_features = torch.matmul(adjacency_matrix, node_features)
        return self.update_mlp(node_features)

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) 
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, adj_matrix, timestamps):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num

        self.gcn_layers = torch.nn.ModuleList([
            TemporalGCNLayer(args['hidden_units'], args['hidden_units']) 
            for _ in range(args['num_gcn_layers'])
        ])
        self.timestamps = timestamps

        self.adj_matrix = adj_matrix
        self.item_emb = torch.nn.Embedding(item_num + 1, args['hidden_units'], padding_idx=0)

        self.layer_agg = None
        
        self.pos_emb = torch.nn.Embedding(args['maxlen']+1, args['hidden_units'], padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args['dropout_rate'])

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args['hidden_units'], eps=1e-8)

        for _ in range(args['num_blocks']):
            new_attn_layernorm = torch.nn.LayerNorm(args['hidden_units'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args['hidden_units'],
                                                          args['num_heads'],
                                                          args['dropout_rate'])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args['hidden_units'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args['hidden_units'], args['dropout_rate'])
            self.forward_layers.append(new_fwd_layer)
            
    def get_graph_embeddings(self):
        layer_output = torch.nn.functional.embedding(torch.arange(self.item_num+1), self.item_emb.weight)

        for gcn_layer in self.gcn_layers:
            layer_output = gcn_layer(layer_output, self.adj_matrix, self.timestamps)

        return layer_output
        
    def augment_sequence(self, log_seqs, dropout_prob=0.2):
        mask = torch.rand(log_seqs.shape) > dropout_prob
        return log_seqs * mask
        
    def log2feats(self, log_seqs, augment=False):
        gcn_embeddings = self.get_graph_embeddings()
        raw_item_embeddings = self.item_emb.weight  

        enhanced_item_embeddings = raw_item_embeddings + gcn_embeddings
        seqs = torch.nn.functional.embedding(torch.LongTensor(log_seqs), enhanced_item_embeddings)

        if augment:
            seqs_aug = self.augment_sequence(seqs)
        else:
            seqs_aug = seqs  
        
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)
        log_feats_aug = self.last_layernorm(seqs_aug)
        
        return log_feats, log_feats_aug

    def contrastive_loss(self, z_i, z_j, temperature=0.5):
        z_i = torch.nn.functional.normalize(z_i, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, dim=-1)

        z_i = z_i.view(128, -1)
        z_j = z_j.view(128, -1) 
        
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
        neg_sim = torch.exp(torch.matmul(z_i, z_j.T) / temperature)
    
        loss = -torch.log(pos_sim / torch.sum(neg_sim, dim=-1))
        return loss.mean()

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, contrastive=True):
        log_feats, log_feats_aug = self.log2feats(log_seqs)
        
        gcn_embeddings = self.get_graph_embeddings()
        enhanced_item_embeddings = self.item_emb.weight + gcn_embeddings 
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        if contrastive:
            cl_loss = self.contrastive_loss(log_feats, log_feats_aug)
        else:
            cl_loss = 0

        return pos_logits, neg_logits, cl_loss 

        return pos_logits, neg_logits 

    def predict(self, user_ids, log_seqs, item_indices): 
        log_feats, _ = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :] 

        gcn_embeddings = self.get_graph_embeddings()
        enhanced_item_embeddings = self.item_emb.weight + gcn_embeddings  

        item_embs = torch.nn.functional.embedding(torch.LongTensor(item_indices), enhanced_item_embeddings)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits