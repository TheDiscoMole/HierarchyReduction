# HierarchyReduction

PyTorch implementation of my Hierarchy Reduction layer. (Tensorflow implementation is easy, just ugly and inefficient due to the static nature of the computational graph)

wtf do I even mean by *information hierarchy reduction*... let me give you a __text based twitter example__ (__NOTE: this can be used for any type of data__):

Let's say you are analyzing a constant stream of Tweets and due to the size of your Net you are accumulating 100 Tweets between every iteration of your Net. For the most part, in such a scenario, your information and network hierarchy would ideally function a little something like this:

<p align="center">
    <img src="https://image.ibb.co/m8sKkU/Twitter_Hierarchy_Example.jpg"/>
</p>

Here we go from character embeddings, to token embeddings, to individual-tweet embeddings and finally to tweet-stream embeddings. This results in the remainder of our DL model now being able to receive a very logical reduction of the meaning vector space.

Neural Network implementations currently can be a little awkward when trying to emulate higher level information hierarchies. Even when using all kinds of ugly padding, reducing information hierarchies more than twice is a massive pain. Going back to the example, just to get to the individual-tweet embedding level you would have to do something like this:

```
import torch.nn as nn

class Model(nn.Module):
    def __init__ (self, character_embedding_size, token_embedding_size, tweet_embedding_size):
        super(Model, self).__init__()

        self.character_reduction = nn.LSTM(
            input_size=character_embedding_size,
            hidden_size=token_embedding_size,
            bidirectional=True)
        self.token_reduction = nn.LSTM(
            input_size=token_embedding_size,
            hidden_size=tweet_embedding_size,
            bidirectional=True)

    def __call__ (self, tweet):
        output,(h,c) = self.character_reduction(tweet)

        token_embeddings = output[-1]                       # [num_words,token_embedding]
        token_embeddings = token_embeddings.unsqueeze_(1)   # [num_words,1,token_embedding]

        output,(h,c) = self.token_reduction(token_embeddings)

        return output[-1]

model = Model()
tweet = model([max_letters,num_words,character_embedding])
```

Clearly this approach is already hitting a brick wall and requires the user to pad each word to `max_letters`, which is quite frankly giving me OCD spasm.

`HierarchyReduction` works around this issue and at the cost of a little efficiency and can even be used to completely remove the need for manual padding from DL models:

```
import torch
import torch.nn as nn

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
```

`HierarchyReduction` uses a cell that can reduce a single batch of inputs down to a single output and uses `slices` to split the input space into n batches.

So in the twitter example you just stack all the character embeddings on top of each other and keep track of the token lengths to build an array of slice indices. Then you build an array of slice indices of the numbers of tokens per tweet and so forth. No matter how high the data hierarchy is you can reduce it all you like, without having to pad a thing. Additionally building these `slices` tensors is more straight forward that handling padding as you will most likely be iterating any input data ingest anyway.
