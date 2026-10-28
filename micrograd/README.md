# 01 · micrograd

Lecture 1 of Andrej Karpathy's Neural Networks: Zero to Hero series,
[The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0).
I rebuilt the scalar autograd engine and the small neural net library from the video, then
trained the net on a real dataset.


`engine.py` holds `Value`, a scalar that records how it was computed and backpropagates
through various operations.

It builds `Neuron`, `Layer` and `MLP` on top
of it. 

## Demo: spotting forged banknotes

The [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
has 1,372 notes labeled genuine or forged. Each note has 4 features taken from an image of it:
wavelet variance, skewness, kurtosis and entropy.

The model is `MLP(4, [8, 8, 1])` with tanh activations, and the labels are encoded as -1 and +1 to match tanh's range. I trained it on 1,098 notes for 500 SGD steps, each on a random batch of
32 with a learning rate of 0.03, minimizing the sum of squared errors. It got all 274 test notes
right.

The two classes are almost linearly separable, so a perfect score on this data is expected
from a small net. All of the gradients come from `engine.py`; 

PyTorch isn't used.

## Running it

Open `demo.ipynb` with the repo's virtual environment and run all cells. The notebook imports
`engine` and `nn` from this folder.
