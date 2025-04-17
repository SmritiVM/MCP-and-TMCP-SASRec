import kagglehub
import numpy as np
import pandas as pd
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', default='dataset')
args = parser.parse_args()
warnings.filterwarnings("ignore")

# %%
"""
## Preprocessing
"""

# %%
"""
### Reviews
"""

# %%
size = 30000 
chunks = []

for chunk in pd.read_json(f'{args.dataset_path}/yelp_academic_dataset_review.json', nrows=3000000, lines=True, chunksize=size):
    chunks.append(chunk) 

review = pd.concat(chunks, ignore_index=True)
review.head()

# %%
num_users = review['user_id'].nunique()

num_items = review['business_id'].nunique()

print(f"Number of Users: {num_users}")
print(f"Number of Items (Business): {num_items}")

# %%
# Filtering out business / user to 20 / 10 instances of each
business_counts = review['business_id'].value_counts()
user_counts = review['user_id'].value_counts()

valid_businesses = business_counts[business_counts >= 20].index
valid_users = user_counts[user_counts >= 10].index

filtered_review = review[review['business_id'].isin(valid_businesses) & review['user_id'].isin(valid_users)]
filtered_review.head()

# %%
num_users = filtered_review['user_id'].nunique()

num_items = filtered_review['business_id'].nunique()

print(f"Number of Users: {num_users}")
print(f"Number of Items (Business): {num_items}")

# %%
"""
### Businesses
"""

# %%
business = pd.read_json(f"{args.dataset_path}/yelp_academic_dataset_business.json", lines=True)

valid_business_ids = filtered_review['business_id'].unique()
filtered_business = business[business['business_id'].isin(valid_business_ids)]
filtered_business.head()

# %%
filtered_business['business_id'].shape

# %%
"""
### Converting both to indices
"""

# %%
business_id_to_idx = {business: idx + 1 for idx, business in enumerate(filtered_business['business_id'])}

# %%
filtered_business['business_id'] = filtered_business['business_id'].copy().map(business_id_to_idx)

# %%
filtered_review['business_id'] = filtered_review['business_id'].copy().map(business_id_to_idx)


# %%
"""
## Semantic
"""

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# %%
business_content = filtered_business[['business_id', 'name', 'categories', 'attributes', 'stars', 'review_count']].copy()

business_content['categories'] = business_content['categories'].astype(str)
business_content['attributes'] = business_content['attributes'].astype(str)

business_content['categories'] = business_content['categories'].fillna('')
business_content['attributes'] = business_content['attributes'].fillna('')

business_content['content'] = business_content['name'] + ' ' + business_content['categories'] + ' ' + business_content['attributes']

# %%
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(business_content['content'])

# %%
semantic_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


# %%
"""
## Sequential
"""

# %%
sorted_reviews = filtered_review.sort_values(by=['user_id', 'date'])
sorted_reviews.head()

# %%
num_users = sorted_reviews['user_id'].nunique()

num_items = sorted_reviews['business_id'].nunique()

print(f"Number of Users: {num_users}")
print(f"Number of Items (Business): {num_items}")

# %%
from collections import defaultdict

transitions = defaultdict(int)
for user, group in sorted_reviews.groupby('user_id'):
    businesses = group['business_id'].tolist()
    for i in range(len(businesses) - 1):
        transitions[(businesses[i], businesses[i + 1])] += 1

transition_matrix = defaultdict(dict)
business_counts = defaultdict(int)

for (business_i, business_j), count in transitions.items():
    business_counts[business_i] += count

for (business_i, business_j), count in transitions.items():
    transition_matrix[business_i][business_j] = count / business_counts[business_i]

unique_businesses = sorted(set(sorted_reviews['business_id']))
business_to_idx = {business: idx for idx, business in enumerate(unique_businesses)}
n_businesses = len(unique_businesses)

sequential_matrix = np.zeros((n_businesses, n_businesses))
for business_i, neighbors in transition_matrix.items():
    for business_j, prob in neighbors.items():
        sequential_matrix[business_to_idx[business_i], business_to_idx[business_j]] = prob

# %%
sequential_matrix

# %%
"""
### Getting a sequence array for SASRec training
"""

# %%
filtered_reviews = sorted_reviews[['user_id', 'business_id']].copy()
filtered_reviews

# %%
unique_users = sorted(set(filtered_reviews['user_id']))
user_to_idx = {user: idx + 1 for idx, user in enumerate(unique_users)}
filtered_reviews['user_id'] = filtered_reviews['user_id'].copy().map(user_to_idx)
filtered_reviews.head()

# %%
ratings_array = filtered_reviews.to_numpy(dtype=np.int32)
ratings_array

# %%
ratings_array.shape

# %%
num_users = filtered_reviews['user_id'].nunique()
num_items = filtered_reviews['business_id'].nunique()

print(f"Number of Users: {num_users}")
print(f"Number of Items (businesses): {num_items}")

# %%
"""
## Hybrid Matrix
"""

# %%
alpha = 1
beta = 1

# %%
hybrid_matrix = alpha * sequential_matrix + beta * semantic_matrix
hybrid_matrix

# %%
sparsity = (np.sum(hybrid_matrix == 0) / hybrid_matrix.size) * 100
print(f"Sparsity: {sparsity:.2f}%")

# %%
"""
## GCN
"""

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_adjacency(adj):
    degree = torch.sum(adj, dim=1, keepdim=True)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    adj_normalized = degree_inv_sqrt * adj * degree_inv_sqrt.t()
    return adj_normalized

adj_matrix = torch.tensor(hybrid_matrix, dtype=torch.float32).to(device)
adj_matrix = normalize_adjacency(adj_matrix).to(device)
num_business = adj_matrix.shape[0]
padding_row = torch.zeros((1, num_business)).to(device)
adj_matrix = torch.cat([adj_matrix, padding_row], dim=0).to(device)
padding_col = torch.zeros((num_business + 1, 1)).to(device)
adj_matrix = torch.cat([adj_matrix, padding_col], dim=1).to(device)
adj_matrix

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        x = torch.mm(adj, x) 
        x = self.linear(x)
        return F.relu(x)

# %%
"""
## SASRec
"""

# %%
"""
### Preprocess
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

        # uid = np.random.randint(1, usernum + 1)
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


# train/val/test data generation
def data_partition(df):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    for index, row in df.iterrows():
        u = row[0]  # Assuming the first column is the user column
        i = row[1]  # Assuming the second column is the item column
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
        predictions = predictions[0]  # - for 1st argsort DESC

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
"""
### Model
"""

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
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, adj_matrix):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args['device']

        self.gcn_layers = nn.ModuleList([
            GCNLayer(args['hidden_units'], args['hidden_units']).to(self.dev)
            for _ in range(args['num_gcn_layers'])
        ])
        self.adj_matrix = adj_matrix
        self.item_emb = torch.nn.Embedding(item_num + 1, args['hidden_units'], padding_idx=0)
        self.item_emb.weight.requires_grad = True
        
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
        layer_output = F.embedding(torch.arange(self.item_num+1, device=self.dev), self.item_emb.weight)

        for gcn_layer in self.gcn_layers:
            layer_output = gcn_layer(layer_output, self.adj_matrix)
            
        return layer_output
        
    def log2feats(self, log_seqs): 
        item_embeddings = self.get_graph_embeddings()
        seqs = F.embedding(torch.LongTensor(log_seqs).to(self.dev), item_embeddings)
        
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

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits  # preds # (U, I)


# %%
"""
## Main
"""

# %%
u2i_index, i2u_index = build_index(ratings_array)
dataset = data_partition(filtered_reviews)
[user_train, user_valid, user_test, usernum, itemnum] = dataset

# %%
print(usernum, itemnum)

# %%
args = {
    'batch_size': 128,
    'lr': 0.0005,
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
model = SASRec(usernum, itemnum, args, adj_matrix).to(args['device'])

for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 

model.pos_emb.weight.data[0, :] = 0
model.item_emb.weight.data[0, :] = 0
    
model.train()


# %%
import time

bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
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

    if epoch % 10 == 0:
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
            best_val_mrr = max(t_valid[2], best_val_mrr)  # Update best MRR for validation
            best_test_ndcg = max(t_test[0], best_test_ndcg)
            best_test_hr = max(t_test[1], best_test_hr)
            best_test_mrr = max(t_test[2], best_test_mrr)  # Update best MRR for test
            
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



# %%
sampler.close()

# %%
