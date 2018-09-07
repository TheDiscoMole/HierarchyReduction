import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchyReduction(nn.Module):
    """
    NN layer that reduces the information hierarchy of the input.

    By slicing the inputs into n batches, and applying a cell that reduces a batch
    into a single lower dimensional output, you can reduce the inforamtional hierarchy
    of your data indefinitely without running into Tensor/ugly-padding limits.

    Args:
        cell: takes an input with shape [old_batches,...] and returns an output with
              shape [...]

    source:
    """
    def __init__ (self, cell):
        super(HierarchyReduction, self).__init__()
        self.cell = cell

    """
    inputs:
        slices: example input [0,9,15, ... len(inputs)]
        inputs: [stached_batches,...]
    outputs:
        output: [un_stacked_baches,...]
    """
    def __call__ (self, slices, inputs):
        outputs = [
            cell( inputs.narrow(0,slices[i],slices[i+1]) )
            for i in range(slices.size()-1)]
        return torch.stack(outputs)
