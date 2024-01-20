import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

eps = 1e-7

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

def tc_loss(x1, x2, x3=None, temp=0.05):
    device = x1.device

    N = x1.shape[0]
    
    loss_fct = nn.CrossEntropyLoss() 
    sim = Similarity(temp=temp)
    
    cos = sim(x1.unsqueeze(1), x2.unsqueeze(0))
    N = x1.shape[0]
    labels = torch.arange(cos.size(0)).long().to(device)

    loss = loss_fct(cos, labels)

    return loss, cos