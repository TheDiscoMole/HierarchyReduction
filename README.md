# HierarchyReduction

Tensorflow and pyTorch implementation of my Hierarchy Reduction layer.

wtf do I even mean by "information hierarchy reduction"... let me give you a text based twitter example (though this can apply to any type of input data):

Let's say you are analyzing a constant stream of Tweets and due to the size of your Net you are accumulating 100 Tweets during every clock of your Net. This data set can be modeled in multiple different ways, like this for example:

<p align="center">
    <img src="https://image.ibb.co/gYyQ9p/Twitter_Hierarchy_Example.jpg"/>
</p>

Neural Network implementations currently can be a little awkward when trying to emulate higher level information hierarchies. Even when using all kinds of ugly padding, reducing information hierarchies more than twice is a massive pain.
