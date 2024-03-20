import json
import collections
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit as masking
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import networkx as nx

# print(torch.__version__)

with open("git_data/musae_git_features.json") as json_data:
    data_raw = json.load(json_data)

edges=pd.read_csv("git_data/musae_git_edges.csv")
target_df=pd.read_csv("git_data/musae_git_target.csv")

# print("5 top nodes")
# print(target_df.head(5).to_markdown())
# print()
# print("5 last nodes")
# print(target_df.tail(5).to_markdown())

feats=[]
for i in range(len(data_raw)):
    feats+=data_raw[str(i)]

def encode_data():
    nodes_included=len(data_raw)

    data_encoded={}
    for i in range(nodes_included):# 
        one_hot_feat=np.array([0]*(max(feats)+1))
        this_feat=data_raw[str(i)]
        one_hot_feat[this_feat]=1
        data_encoded[str(i)]=list(one_hot_feat)

    return(data_encoded, None)


def construct_graph(data_encoded):
    node_features_list=list(data_encoded.values())
    node_features=torch.tensor(node_features_list)
    node_labels=torch.tensor(target_df['ml_target'].values)
    edge_index = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
    data = Data(x=node_features, y=node_labels, edge_index=edge_index, num_classes = 2)

    return(data)

data_encoded,_ = encode_data()
data = construct_graph(data_encoded=data_encoded)

# with open("reddit_dataset/reddit_features.json") as json_data:
#     features = json.load(json_data)

# edges=pd.read_csv("reddit_dataset/reddit_edges.csv")
# targets=pd.read_csv("reddit_dataset/reddit_targets.csv")

# print("5 top nodes")
# print(targets.head(5).to_markdown())
# print()
# print("5 last nodes")
# print(targets.tail(5).to_markdown())

# # Prepare the data for PyTorch Geometric
# edge_index = torch.tensor(edges.values, dtype=torch.long).t().contiguous()

# x = torch.tensor([features[str(i)] for i in range(len(features))], dtype=torch.float)
# y = torch.tensor(targets['target'].values, dtype=torch.long)

# data = Data(x=x, edge_index=edge_index, y=y, num_classes=2)

print(f'Dataset: {data}:')
print('======================')
print(f'Number of features: {data.num_features}')
print(f'Number of classes: {data.num_classes}')


msk = masking(split="train_rest", num_splits = 1, num_val = 0.2, num_test= 0.2)

data = msk(data)
print(data)
print()
print("training samples",torch.sum(data.train_mask).item())
print("validation samples",torch.sum(data.val_mask ).item())
print("test samples",torch.sum(data.test_mask ).item())


# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATConv(data.num_features, hidden_channels,heads)
        self.conv2 = GATConv(heads*hidden_channels, data.num_classes,heads)

    def forward(self, data):
        x = data.x.float()
        edge_index =  data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def runGNN(hidden_channels, heads, epochs, lr):
    
    model = GCN(hidden_channels, heads)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)  # Define optimizer.

    def train(mask):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def test(mask):
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)# Use the class with highest probability.
        correct = pred[mask] == data.y[mask] # Check against ground-truth labels.
        acc = int(correct.sum()) / int(mask.sum()) # Derive ratio of correct predictions.
        return acc

    train_loss_all = []
    val_loss_all = []
    test_loss_all = []


    train_acc_all = []
    val_acc_all = []
    test_acc_all = []

    best_acc = 0.0

    for epoch in range (epochs+1):
        train_loss = train(data.train_mask)
        train_loss_all.append(train_loss.detach().numpy())

        val_loss = train(data.val_mask)
        val_loss_all.append(val_loss.detach().numpy())

        test_loss = train(data.test_mask)
        test_loss_all.append(test_loss.detach().numpy())

        train_acc = test(data.train_mask)
        train_acc_all.append(train_acc)

        val_acc = test(data.val_mask)
        val_acc_all.append(val_acc)

        test_acc = test(data.test_mask)
        test_acc_all.append(test_acc)

        if np.round(val_acc, 4)> np.round(best_acc, 4):
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            best_acc=val_acc

    # Test the model
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))

    return train_loss_all, val_loss_all, test_loss_all, train_acc_all, val_acc_all, test_acc_all


train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = runGNN(hidden_channels = 8, heads = 8, epochs = 20, lr=0.005)

plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label='Train loss', c='green')
plt.plot(np.arange(1, len(val_loss) + 1), val_loss, label='Valodation loss', c='purple')
plt.plot(np.arange(1, len(test_loss) + 1), test_loss, label='Testing loss', c='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('GATConv')
plt.legend(loc='upper right', fontsize='x-large')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(train_acc) + 1), train_acc, label='Train accuracy', c='yellow')
plt.plot(np.arange(1, len(val_acc) + 1), val_acc, label='Validation accuracy', c='blue')
plt.plot(np.arange(1, len(test_acc) + 1), test_acc, label='Testing accuracy', c='red')
plt.xlabel('Epochs')
plt.ylabel('Accurarcy')
plt.title('GATConv')
plt.legend(loc='lower right', fontsize='x-large')
plt.show()

