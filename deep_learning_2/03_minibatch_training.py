# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# -

#export
from exp.nb_02 import *
import torch.nn.functional as F

# ## Initial setup

# ### Data

mpl.rcParams['image.cmap'] = 'gray'

x_train,y_train,x_valid,y_valid = get_data()

n,m = x_train.shape
c = y_train.max()+1
nh = 50


# ### Cross entropy loss

class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]
        
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x


model = Model(m, nh, 10)

pred = model(x_train)


def log_softmax(x): return torch.log(x.exp()/(x.exp().sum(-1,keepdim=True)))


def nll(input, target): return -input[range(target.shape[0]), target].mean()


loss = nll(log_softmax(pred), y_train)


def log_softmax(x): return x.sub_(x.exp().sum(-1,keepdim=True).log())


test_near(nll(log_softmax(pred), y_train), loss)


def log_softmax(x): return x.sub_(x.logsumexp(-1,keepdim=True))


test_near(nll(log_softmax(pred), y_train), loss)

test_near(F.nll_loss(F.log_softmax(pred, -1), y_train), loss)

test_near(F.cross_entropy(pred, y_train), loss)

# ## Basic training loop

loss_func = F.cross_entropy


#export
def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()


# +
bs=64                  # batch size

xb = x_train[0:bs]     # a mini-batch from x
preds = model(xb)      # predictions
preds[0], preds.shape
