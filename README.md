## HierarchyReduction

This repository proposes Informational Hierarchy Reduction a potential alternative approach to increasing the context window of LLMs. **NOTE:** fully theoretical, basically useless without relevant optimizations

```py
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
```

By keeping track of sequence length indices of our input data, at different levels of their informational hierarchy, we can create a repeatable reduction procedure that allows for arbitrarily long and complex input sequences to be appropriately reduced to a single feature vector.

The following is a pseudocode example:
```py
word_embeddings = HierarchyReduction(character_embeddings, [(0,4), (4,9), ...])
sentence_embeddings = HierarchyReduction(word_embeddings, [(0,10), (10,22), ...])
paragraph_embeddings = HierarchyReduction(sentence_embeddings, [(0,32), (32,50), ...])
...
```

This example process could be arbitrarily extended until an entire corpus of data is coherently embedded as a single vector. At each level the sequences for the next level are stacked along the batch dimension. This should, at least during inference, enable us to write rich encoders to apply at each level without having to worry about memory constrains or compute. Given the popular Transformer architecture we would be able to write a more coherent representation of positional encodings, as they would incrementally represent position across their individual levels of informational hierarchy.

Currently the most sensible way to implement this behaviour is to utilize `nested_tensor`, but unfortunately due to its beta status there are several impeding factor toward the efficiency of this method:

- `transpose` and similar operations require contiguous copies of our inputs
- `sum` (and hopefully as some point `mean`) cannot fully reduce a `nested_tensor` into a normal tensor
