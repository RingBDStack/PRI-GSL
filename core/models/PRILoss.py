
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse

def incidence(g1, device):
    '''
    get incidence matrix
    '''
    E = nx.incidence_matrix(g1, oriented=True)
    E = E.todense()
    return torch.from_numpy(E).to(device)

def vn_entropy(k, eps=1e-8):

    k2 = k/torch.trace(k)  # normalization
    # k2 = k

    # eigv = torch.abs(torch.symeig(k2, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigvalsh(k2, UPLO='U'))
    entropy = - torch.sum(eigv[eigv>0]*torch.log(eigv[eigv>0]+eps))
    return entropy


def entropy_loss(sigma, rho, beta, alpha, entropy_fn = vn_entropy):
    assert(beta>=0), "beta shall be >=0"
    if beta > 0:
        loss = 0.5*(1-beta)/beta * entropy_fn(sigma) + entropy_fn(0.5 * (sigma+rho))
        return loss #- connectivity_loss 
    else:
        return entropy_fn(sigma) - conectivity_loss 

def pri(syn_data: list, data: list, device='cuda'):
    pri_loss = 0.0
    for syn_graph, real_graph in zip(syn_data, data):
        syn_graph = to_networkx(syn_graph)
        real_graph = to_networkx(real_graph)

        real_E = incidence(real_graph, device=device)
        syn_E = incidence(syn_graph, device=device)

        sigma = syn_E@syn_E.T
        rho = (real_E@real_E.T)[:len(sigma), :len(sigma)]
        cur_loss = entropy_loss(sigma, rho, 0, 0.01)
        # print(cur_loss.detach())
        pri_loss += cur_loss
    return pri_loss/len(syn_data)

def pri_node(syn_data, data, rnd_index, adj_knn, device='cuda', beta=0.1):
    
    graph2L = lambda data: torch.diag(degree(dense_to_sparse(data)[0][0], num_nodes=len(data))) - data
    sigma = graph2L(syn_data).squeeze()
    rho = graph2L(data).squeeze()

    cur_loss = entropy_loss(sigma, rho, beta=beta, alpha=0.01)
    return cur_loss

