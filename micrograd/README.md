# 01 · micrograd

Lecture 1 of Andrej Karpathy's Neural Networks: Zero to Hero series,
[The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0).
I rebuilt the scalar autograd engine and the small neural net library from the video, then
trained the net on a real dataset.

## Files

| File | Contents |
|---|---|
| `engine.py` | `Value`, a scalar that records how it was computed and backpropagates through `+`, `*`, `-`, `/`, `**` and `tanh` |
| `nn.py` | `Neuron`, `Layer` and `MLP`, built on `Value` |
| `data/data.ipynb` | Downloads the dataset, shuffles it, splits it 80/20 and normalizes the features with the training set's mean and std |
| `data/banknote_split.json` | The normalized split, plus the means and stds |
| `demo.ipynb` | Trains the MLP and checks it against the test set |

## Demo: spotting forged banknotes

The [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
has 1,372 notes labeled genuine or forged. Each note has 4 features taken from an image of it:
wavelet variance, skewness, kurtosis and entropy.

| Setting | Value |
|---|---|
| Model | `MLP(4, [8, 8, 1])` with tanh, labels encoded as -1 / +1 |
| Loss | Sum of squared errors over each batch |
| Training | 500 SGD steps, random batches of 32, learning rate 0.03 |
| Split | 1,098 train / 274 test |
| Test accuracy | 100% (274 of 274) |

The two classes are almost linearly separable, so a perfect score on this data is expected
from a small net. All of the gradients come from `engine.py`; PyTorch isn't used.

## Running it

Open `demo.ipynb` with the repo's virtual environment and run all cells. The notebook imports
`engine` and `nn` from this folder.
