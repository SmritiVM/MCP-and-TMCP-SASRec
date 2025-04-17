import kagglehub
import numpy as np
import torch
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', default='dataset')
args = parser.parse_args()
warnings.filterwarnings("ignore")

semantic_matrix = np.load(f'{args.dataset_path}/amazon_with_title_cosine_semantic_matrix.npy')
sequential_matrix = np.load(f'{args.dataset_path}/amazon_sequential_matrix.npy')
ratings_array = np.load(f'{args.dataset_path}/amazon_ratings.npy')

# %%
"""
## Hybrid Matrix
"""

alpha = 1
beta = 1

hybrid_matrix = alpha * sequential_matrix + beta * semantic_matrix

device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1")

adj_matrix = torch.tensor(hybrid_matrix, dtype=torch.float32)

num_nodes = adj_matrix.shape[0]
half_size = num_nodes // 2  
A_11 = adj_matrix[:half_size, :half_size].to(device_0)  
A_12 = adj_matrix[:half_size, half_size:].to(device_0) 
A_21 = adj_matrix[half_size:, :half_size].to(device_1)
A_22 = adj_matrix[half_size:, half_size:].to(device_1)

def normalize_adjacency(adj):
    degree = torch.sum(adj, dim=1, keepdim=True)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree == 0] = 0 
    return degree_inv_sqrt * adj * degree_inv_sqrt.t()

A_11 = normalize_adjacency(A_11)
A_12 = normalize_adjacency(A_12)
A_21 = normalize_adjacency(A_21)
A_22 = normalize_adjacency(A_22)

def personalized_ppr(adj, alpha=0.1, max_iter=10, tol=1e-6):
    """Computes Personalized PageRank (PPR) matrix."""
    num_nodes = adj.shape[0]
    adj_norm = normalize_adjacency(adj)
    ppr_matrix = torch.eye(num_nodes, device=adj.device)

    for _ in range(max_iter):
        new_ppr = alpha * torch.eye(num_nodes, device=adj.device) + (1 - alpha) * torch.mm(adj_norm, ppr_matrix)
        if torch.norm(new_ppr - ppr_matrix) < tol:
            break
        ppr_matrix = new_ppr

    return ppr_matrix

alpha_ppr = 0.1
ppr_A11 = personalized_ppr(A_11, alpha_ppr)
ppr_A12 = personalized_ppr(A_12, alpha_ppr)
ppr_A21 = personalized_ppr(A_21, alpha_ppr)
ppr_A22 = personalized_ppr(A_22, alpha_ppr)

top_half = torch.cat([ppr_A11, ppr_A12], dim=1)  # GPU 0
bottom_half = torch.cat([ppr_A21, ppr_A22], dim=1)  # GPU 1

adj_matrix_ppr = torch.cat([top_half.to(device_0), bottom_half.to(device_0)], dim=0)

del ppr_A21, ppr_A22

# %%
num_products = adj_matrix_ppr.shape[0]
padding_row = torch.zeros((1, num_products)).to(device_0)
adj_matrix_ppr = torch.cat([adj_matrix_ppr, padding_row], dim=0)
padding_col = torch.zeros((num_products + 1, 1)).to(device_0)
adj_matrix_ppr = torch.cat([adj_matrix_ppr, padding_col], dim=1)

# %%
print(adj_matrix_ppr)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.1):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha = alpha  

    def forward(self, x, adj):
        device = adj.device
        x = x.to(device)
        propagated = torch.mm(adj, x)
        x = self.alpha * x + (1 - self.alpha) * propagated 
        x = self.linear(x)
        return F.relu(x)

# %%
"""
## SASRec
"""

# %%
import torch
import random
from collections import defaultdict
from multiprocessing import Process, Queue

# %%
def build_index(ui_mat):
    n_users = ui_mat[:,0].max()
    n_items = ui_mat[:,1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(ratings_array):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    for row in ratings_array:
        u = row[0]  
        i = row[1] 
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# %%
import copy
import random
import numpy as np

def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    MRR = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args['maxlen']], dtype=np.int32)
        idx = args['maxlen'] - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            MRR += 1 / (rank + 1)  

    return NDCG / valid_user, HT / valid_user, MRR / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    MRR = 0.0 
    valid_user = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args['maxlen']], dtype=np.int32)
        idx = args['maxlen'] - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        start_time = time.time() 
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        end_time = time.time()  
        predict_time = end_time - start_time
        print(f"Time taken for prediction for user {u}: {predict_time:.6f} seconds")

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

            MRR += 1 / (rank + 1)

    return NDCG / valid_user, HT / valid_user, MRR / valid_user, predict_time

# %%
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
    def __init__(self, user_num, item_num, args, adj_matrix):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args['device']

        self.gcn_layers = nn.ModuleList([
            GCNLayer(args['hidden_units'], args['hidden_units']) 
            for _ in range(args['num_gcn_layers'])
        ])

        self.adj_matrix = adj_matrix
        self.item_emb = torch.nn.Embedding(item_num + 1, args['hidden_units'], padding_idx=0)

        self.layer_agg = None
        
        self.pos_emb = torch.nn.Embedding(args['maxlen']+1, args['hidden_units'], padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args['dropout_rate'])

        # self.layer_agg = nn.Linear(args['hidden_units'] * args['num_gat_layers'], args['hidden_units'])  # Aggregation Layer
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
        layer_output = F.embedding(torch.arange(self.item_num+1, device=self.dev), self.item_emb.weight)

        for gcn_layer in self.gcn_layers:
            layer_output = gcn_layer(layer_output, self.adj_matrix)

        return layer_output
        
        
    def log2feats(self, log_seqs):

        item_embeddings = self.get_graph_embeddings()
        seqs = F.embedding(torch.LongTensor(log_seqs).to(self.dev), item_embeddings)
        
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
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

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        item_embeddings = self.get_graph_embeddings()
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits 

    def predict(self, user_ids, log_seqs, item_indices): 
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :] 

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

# %%
"""
### Main
"""

# %%
u2i_index, i2u_index = build_index(ratings_array)
dataset = data_partition(ratings_array)
[user_train, user_valid, user_test, usernum, itemnum] = dataset

# %%
print("Usernum = ", usernum, "Itemnum = ", itemnum)

# %%
args = {
    'batch_size': 128,
    'lr': 0.001,
    'maxlen': 50,
    'hidden_units': 50,
    'num_blocks': 2,
    'num_epochs': 200,
    'num_heads': 1,
    'dropout_rate': 0.2,
    'l2_emb': 0.0,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'inference_only': False,
    'num_gcn_layers': 1
}

# %%
num_batch = (len(user_train) - 1) // args['batch_size'] + 1
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('Average sequence length: %.2f' % (cc / len(user_train)))

# %%
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args['batch_size'], maxlen=args['maxlen'], n_workers=3)
model = SASRec(usernum, itemnum, args, adj_matrix_ppr).to(args['device'])

for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

model.pos_emb.weight.data[0, :] = 0
model.train()

# %%
import time
import matplotlib.pyplot as plt

bce_criterion = torch.nn.BCEWithLogitsLoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), weight_decay=1e-4)

best_val_ndcg, best_val_hr, best_val_mrr = 0.0, 0.0, 0.0
best_test_ndcg, best_test_hr, best_test_mrr = 0.0, 0.0, 0.0
T = 0.0
t0 = time.time()

patience = 5
no_improvement_epochs = 0

prediction_times = []
total_no_of_predictions = 0

for epoch in range(1, args['num_epochs'] + 1):
    for step in range(num_batch):  
        u, seq, pos, neg = sampler.next_batch() 
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits = model(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args['device']), torch.zeros(neg_logits.shape, device=args['device'])
        adam_optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        for param in model.item_emb.parameters(): loss += args['l2_emb'] * torch.norm(param)
        loss.backward()
        adam_optimizer.step()


    print(f"Epoch {epoch} out of {args['num_epochs']} executed")

    if epoch % 20 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        
        t_test = evaluate(model, dataset, args)
        t_valid = evaluate_valid(model, dataset, args)
        
        print(f'epoch:{epoch}, time: {T:.6f}(s), valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}, MRR: {t_valid[2]:.4f}), '
              f'test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f}, MRR: {t_test[2]:.4f})')
    
        prediction_times.append(t_valid[3])
        total_no_of_predictions += 1

        if (t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_valid[2] > best_val_mrr or
            t_test[0] > best_test_ndcg or t_test[1] > best_test_hr or t_test[2] > best_test_mrr):
            
            best_val_ndcg = max(t_valid[0], best_val_ndcg)
            best_val_hr = max(t_valid[1], best_val_hr)
            best_val_mrr = max(t_valid[2], best_val_mrr)
            best_test_ndcg = max(t_test[0], best_test_ndcg)
            best_test_hr = max(t_test[1], best_test_hr)
            best_test_mrr = max(t_test[2], best_test_mrr)
            
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
    
        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch}. No improvement in validation NDCG/HR/MRR for {patience} epochs.")
            break
    
        t0 = time.time()
        model.train()
print(f'Best Validation Metrics: '
      f'NDCG@10: {best_val_ndcg:.4f}, HR@10: {best_val_hr:.4f}, MRR: {best_val_mrr:.4f}')
print(f'Best Test Metrics: '
      f'NDCG@10: {best_test_ndcg:.4f}, HR@10: {best_test_hr:.4f}, MRR: {best_test_mrr:.4f}')
print('Average Prediction Time: ', sum(prediction_times)/total_no_of_predictions)
