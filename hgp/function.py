r"""
Some pytorch functions probably used in the project.
"""
import torch

class ScaleEstimatorForLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save the input for backward pass
        ctx.save_for_backward(input)
        # Return the hard thresholded output
        input[input < 0] = 0
        input[input > 1] = 1
        return input.float()

    @staticmethod
    def backward(ctx, grad_output):
        
        input, = ctx.saved_tensors
        # 计算每一列不为0的个数获得一个Tensor
        num_nodes = input.size(0) - (input == 0).sum(dim=0)

        input = torch.where(input != 0, 1 / input, torch.tensor(0.0))
        # grad_input为每一个input的倒数
        grad_input = grad_output / num_nodes * input
        return grad_input

class StraightThroughEstimator(torch.autograd.Function):
    # fmt: off
    r"""
    直通估计器.
    ---
    直通估计器是一种用于硬阈值的梯度估计器.  
    
    Args:
        ``input`` (``torch.Tensor``): 输入张量.
    """
    # fmt: on
    
    @staticmethod
    def forward(ctx, input):
        # Save the input for backward pass
        ctx.save_for_backward(input)
        # Return the hard thresholded output
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input
        input, = ctx.saved_tensors
        # Compute the gradient of the soft threshold
        grad_input = grad_output.clone()
        # Pass the gradient through where the input was within the unit interval
        grad_input[input <= 0] = 0
        grad_input[input >= 1] = 0
        return grad_input
    
class StraightThroughEstimator_1(torch.autograd.Function):
    # fmt: off
    r"""
    直通估计器.
    ---
    直通估计器是一种用于硬阈值的梯度估计器.  
    
    Args:
        ``input`` (``torch.Tensor``): 输入张量.
    """
    # fmt: on
    
    @staticmethod
    def forward(ctx, input):
        # Save the input for backward pass
        ctx.save_for_backward(input)
        # 每一行最大值的位置
        input_mask = torch.argmax(input, dim=1)
        # 最大值位置是1，其余是0
        result = torch.zeros_like(input)
        result.scatter_(1, input_mask.unsqueeze(1), 1)
        # Return the hard thresholded output
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input
        input, = ctx.saved_tensors
        # Compute the gradient of the soft threshold
        grad_input = grad_output.clone()
        # Pass the gradient through where the input was within the unit interval
        grad_input[input <= 0] = 0
        grad_input[input >= 1] = 0
        return grad_input
    
class Scale(torch.autograd.Function):
    # fmt: ofF
    r"""
    缩放函数.
    ---
    缩放函数定义了一个缩放因子，用于缩放输入张量.  
    但是，缩放函数的反向传播不会返回缩放因子的梯度.
        
    Args:
        ``scale_factor`` (``float``): 缩放因子.
    """
    # fmt: on
    
    @staticmethod
    def forward(ctx, input, scale_factor):
        ctx.scale_factor = scale_factor
        return input * scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        scale_factor = ctx.scale_factor
        return grad_output * 10, None