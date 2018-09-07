# HierarchyReduction

Tensorflow and pyTorch implementation of my Hierarchy Reduction layer.

wtf do I even mean by "information hierarchy reduction"... let me give you a text based twitter example (though this can apply to any type of input data):

Let's say you are analyzing a constant stream of Tweets and due to the size of your Net you are accumulating 100 Tweets between every iteration of your Net. For the most part, in such a scenario, your information and network hierarchy would ideally function a little something like this:

<p align="center">
    <img src="https://image.ibb.co/gYyQ9p/Twitter_Hierarchy_Example.jpg"/>
</p>

Here we go from character embeddings, to token embeddings, to tweet embeddings and finally to tweet-stream embeddings. And this results in the remainder of our DL model now receiving very logical reduction of the meaning vector space.

Neural Network implementations currently can be a little awkward when trying to emulate higher level information hierarchies. Even when using all kinds of ugly padding, reducing information hierarchies more than twice is a massive pain. Going back to the example, just to get to the individual tweet embedding level you would have to do something like this:

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
