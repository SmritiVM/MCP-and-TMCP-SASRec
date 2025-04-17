import torch
import numpy as np

class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        x = torch.mm(adj, x) 
        x = self.linear(x)
        return torch.nn.functional.relu(x)
    
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
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, adj_matrix):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args['device']

        self.gcn_layers = torch.nn.ModuleList([
            GCNLayer(args['hidden_units'], args['hidden_units']) 
            for _ in range(args['num_gcn_layers'])
        ])
        self.adj_matrix = adj_matrix
        self.item_emb = torch.nn.Embedding(item_num + 1, args['hidden_units'], padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args['maxlen']+1, args['hidden_units'], padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args['dropout_rate'])

        self.layer_agg = None
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
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
        embeddings = [self.item_emb.weight]
        layer_output = embeddings[0]

        for gcn_layer in self.gcn_layers:
            layer_output = gcn_layer(layer_output, self.adj_matrix)
            embeddings.append(layer_output)

        if self.layer_agg == 'sum':
            return torch.stack(embeddings).sum(dim=0)
        elif self.layer_agg == 'avg':
            return torch.stack(embeddings).mean(dim=0)
        elif self.layer_agg == 'concat':
            return torch.cat(embeddings, dim=-1)
        else: 
            return layer_output
   
    def log2feats(self, log_seqs):  
        item_embeddings = self.get_graph_embeddings()
        seqs = torch.nn.functional.embedding(torch.LongTensor(log_seqs).to(self.dev), item_embeddings)
        
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        item_embeddings = self.get_graph_embeddings()
        pos_embs = torch.nn.functional.embedding(torch.LongTensor(pos_seqs).to(self.dev), item_embeddings)
        neg_embs = torch.nn.functional.embedding(torch.LongTensor(neg_seqs).to(self.dev), item_embeddings)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): 
        log_feats = self.log2feats(log_seqs)  

        final_feat = log_feats[:, -1, :]  

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits 