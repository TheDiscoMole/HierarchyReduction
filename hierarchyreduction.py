import torch

class HierarchyReduction1d (torch.nn.Module):
    """
    This module reduces the information hierarchy of the input. By slicing the
    inputs into n batches, and applying a cell that reduces a batch-slice into a
    single lower-dimensional output, we can reduce the informational hierarchy
    of your data "indefinitely".

    args:
    - reducer: reduces the current information level                            (optional|default: torch.sum)

    source: https://github.com/TheDiscoMole/HierarchyReduction
    """
    def __init__ (self, reducer = lambda input: torch.sum(input, dim=-1, keepdim=True)):
        torch.nn.Module.__init__(self)

        # reducer
        self.reducer = reducer

    def forward (self, input, slices):
        # reduce current hierarchy level
        reduction = input.transpose(1,2).contiguous()
        reduction = self.reducer(reduction)
        reduction = torch.stack(reduction.unbind())

        # extract next hierarchy level
        level = reduction.squeeze(-1)
        level = [reduction[i:j] for i,j in slices]
        level = torch.nested.nested_tensor(level)

        return level
