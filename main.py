"""
attributes

ndim -> no of dimensions(basically the number of brackets in the tensor representation)
shape -> shape of tensor
dtype -> Default value depends on device architecture, if GPU is present or not
__len__() -> returns length along the first dimesion / no of elements in the outermost dimension

torch.Tensor:
    size() -> no of elements in every dimension, starting from the outermost
    item() -> output the data stored in the tensor as a Python scalar rather than a tensor object
    rand(size),_like -> creates a tensor of float values 
    tensor(data,dype,device,requires_grad)
    randint(low,high,size), _like(tensor,high)
    ones,zeros, _like(tensor)
    reshape(tensor,size) -> creates new memory locations
    permute(dim:size) -> shares the same memory as original tensor
    view(shape) -> has the same underlyign data as the original tensor
    squeeze(shape)
    unsqueeze(tensor,dim) -> adds a single dimension along dim
    eq() -> compares two tensors of same size element wise
    complex(real=,imag=) -> tensor inputs, must float16/32/64   
    empty(size) -> uses uninitialised memory location to create a tensor, containing numbers in those locations
    linspace(start,end,size) >> creates 1d tensor elements between start and end(inclusive), with equal intervals b/w each element
    
torch.nn
    -> Module - a pre-built framework with built in frameworks for different operations
              - different modules hav differnet functionalities which are then clubbed together
"""

""""
Random findings

*
min/max(tensor,tensor) to compare and create tensor with least vales b/w the two
view() shares the same underlying data as the orginal while reshape() might create new memory allocations for a new array
data and shape information are stored separately
"""

import torch

x = torch.tensor([1,2,3])
print(x)