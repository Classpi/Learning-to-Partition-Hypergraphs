import torch
from torch.types import _device


def loss_bs(outs, hg, device: _device | str):
    # fmt: off
    r"""
    CUBO问题推广至超图范围的损失函数.
    ---
    Args:
        ``outs`` (``torch.nn.Module``): 模型的输出. Size :math:`(N, nums_classes)`.   
        ``hg`` (``Hypergraph``): 超图对象.  
        ''device'' (``torch.device``): 设备.
    """
    # fmt: on
    loss_1 = torch.zeros(outs.shape[1]).to(device)  # loss_1 延续之前的定义
    edges, _ = hg.e
    for vertices in edges:
        vertices = list(vertices)
        loss_1 = loss_1 + (
            torch.sum(outs[vertices], dim=0) / len(vertices)
            - torch.prod(outs[vertices], dim=0)
        )

    loss_1 = 1e-4 * loss_1.sum()
    loss_2 = torch.var(torch.sum(outs, dim=0)).to(device)

    total_loss = loss_1 + loss_2

    return total_loss, loss_1, loss_2


def loss_bs_matrix(outs, hg, device: _device | str):
    # fmt: off
    r"""
    对于超图的损失函数的矩阵形式.
    
        1.计算与顶点``vₙ``处于不同partition的顶点在超边``eₖ``上的数量``ne_k``.  
        2.计算与顶点``vₙ``是否处于该超边``eₖ``上.  
        3.若在,则说明``vₙ``所在的边为 **cut** , 记录该边的损失.  
    
    Args:
        ``outs``(`torch.nn.Module`):  模型的输出. Size :math:`(N, nums_classes)`.   
        ``hg``(`Hypergraph`):  超图对象.  
    """
    # fmt: on
    H = hg.H.to_dense().to(device)
    outs = outs.to(device)
    nn = torch.matmul(outs, (1 - torch.transpose(outs, 0, 1)))
    ne_k = torch.matmul(nn, H)
    ne_k = ne_k.mul(H)

    H_degree = torch.sum(H, dim=0)
    H_degree = H_degree - 0.4

    H_1 = ne_k / H_degree
    a2 = 1 - H_1
    a3 = torch.prod(a2, dim=0)
    a3 = a3.sum()
    loss_1 = -1 * a3

    # pun = torch.mul(ne_k, H)

    # loss_1 = pun.sum()
    loss_2 = torch.var(torch.sum(outs, dim=0)).to(device)

    loss = 50 * loss_1 + loss_2
    return loss, loss_1, loss_2

def loss_bs_matrix_x(outs, adj, device: _device | str):
    
    outs = outs.to(device)
    nn = torch.matmul(outs, (1 - torch.transpose(outs, 0, 1)))
    ne_k = torch.matmul(nn, adj)

    pun = torch.mul(ne_k, adj)

    loss_1 = pun.sum()
    loss_2 = torch.var(torch.sum(outs, dim=0)).to(device)

    loss = loss_1 + loss_2
    return loss, loss_1, loss_2

def loss_bs_matrix_mega(outs, hg, de: torch.Tensor, device: _device | str):
    # fmt: off
    r"""
    对于超图的损失函数的矩阵形式.
    
    过程:
        1. 计算与顶点``vₙ``处于不同partition的顶点在超边``eₖ``上的数量``ne_k``.  
        2. 计算与顶点``vₙ``是否处于该超边``eₖ``上.  
        3. 若在,则说明``vₙ``所在的边为 **cut** , 记录该边的损失.  
        4. 归一化
    
    Args:
        ``outs``(`torch.nn.Module`):  模型的输出. Size :math:`(N, nums_classes)`.   
        ``hg``(`Hypergraph`):  超图对象.  
    """
    # fmt: on
    H = hg.H.to_dense().to(device)
    outs = outs.to(device)
    nn = torch.matmul(outs, (1 - torch.transpose(outs, 0, 1)))
    ne_k = torch.matmul(nn, H)

    pun = torch.mul(ne_k, H)
    pun = pun.sum(dim=0)
    pun = pun.mul(de.sqrt()) 
    loss_1 = pun.sum()
    loss_2 = torch.var(torch.sum(outs, dim=0)).to(device)

    loss = loss_1 + loss_2
    return loss, loss_1, loss_2
