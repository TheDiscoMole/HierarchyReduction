
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchyEncoder (nn.Module):
    """
    NN layer that reduces the information hierarchy of the input. By slicing the
    inputs into n batches, and applying a cell that reduces a batch into a
    single lower-dimensional output, you can reduce the informational hierarchy
    of your data "indefinitely".

    args:
    - cell: takes an input with shape [1,sequence,<dimensions>] and returns an
            output with shape [1,<dimensions>]

    source: https://github.com/TheDiscoMole/HierarchyReduction
    """
    def __init__ (self,cell):
        super(HierarchyReduction,self).__init__()
        self.cell = cell

    """
    inputs:
    - slices: [slice,index]
    - inputs: [1,sequence,<dimensions>]
    outputs:
    - output: [1,sequence,<dimensions>]
    """
    def forward (self,slices,inputs):
        return torch.stack([
            cell( inputs.narrow(0,slices[i][0],slices[i][1]) )
            for i in range(slices.size()-1)])
