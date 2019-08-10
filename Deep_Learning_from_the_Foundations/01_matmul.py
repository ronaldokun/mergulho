# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"lang": "en"}
# ## Matrix multiplication from foundations

# %% [markdown] {"lang": "pt"}
# ## Multiplicação de matrizes somente com o fundamental

# %% [markdown] {"lang": "en"}
# The *foundations* we'll assume throughout this course are:
#
# - Python
# - Python modules (non-DL)
# - pytorch indexable tensor, and tensor creation (including RNGs)
# - fastai.datasets

# %% [markdown] {"lang": "pt"}
# Os *fundamentos* que assumimos ao longo deste curso são:
#
# - Python
# - módulos Python (Exceto DL)
# - tensor indexável do pytorch e criação de tensores (incluindo Gerador de Números Pseudo-Aleatórios)
# - fastai.datasets

# %% [markdown] {"lang": "en"}
# ## Check imports

# %% [markdown] {"lang": "pt"}
# ## Importações

# %%
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

# %%
#export
from exp.nb_00 import *
import operator

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq,'==')


# %%
test_eq(TEST,'test')

# %%
# To run tests in console:
# # ! python run_notebook.py 01_matmul.ipynb

# %% [markdown]
# ## Get data

# %%
#export
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

# %%
path = datasets.download_data(MNIST_URL, ext='.gz'); path

# %%
with gzip.open(path, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

# %%
x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
n,c = x_train.shape
x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max()

# %%
assert x_train.shape[0]==y_train.shape[0]==50000
test_eq(x_train.shape[1],28*28)
test_eq(y_train.min(),0)
test_eq(y_train.max(),9)

# %%
mpl.rcParams['image.cmap'] = 'gray'

# %%
img = x_train[0]

# %%
img.view(28,28).type()

# %%
plt.imshow(img.view((28,28)));

# %% [markdown]
# ## Initial python model

# %%
weights = torch.randn(784,10)/math.sqrt(784)

# %%
bias = torch.zeros(10)


# %% [markdown]
# #### Matrix multiplication

# %%
def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                c[i,j] += a[i,k] * b[k,j]
    return c


# %%
m1 = x_valid[:5]
m2 = weights

# %%
m1.shape,m2.shape

# %%
# %time t1=matmul(m1, m2)

# %%
len(x_train)

# %% [markdown]
# #### Elementwise ops

# %% [markdown]
# Operators (+,-,\*,/,>,<,==) are usually element-wise.
#
# Examples of element-wise operations:

# %%
a = tensor([10., 6, -4])
b = tensor([2., 8, 7])
a,b

# %%
a + b

# %%
(a < b).float().mean()

# %%
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]]); m

# %% [markdown] {"lang": "en"}
# Frobenius norm:
#
# $\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$
#
# *Hint*: you don't normally need to write equations in LaTeX yourself, instead, you can click 'edit' in Wikipedia and copy the LaTeX from there (which is what I did for the above equation). Or on arxiv.org, click "Download: Other formats" in the top right, then "Download source"; rename the downloaded file to end in `.tgz` if it doesn't already, and you should find the source there, including the equations to copy and paste.

# %% [markdown] {"lang": "pt"}
# Norma Frobenius:
#
# $\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$
#
# *Sugestão*: você não precisa escrever equações no LaTeX, você pode clicar em 'editar' na Wikipedia e copiar o LaTeX de lá (o que eu fiz para a equação acima). Ou em arxiv.org, clique em "Download: Outros formatos" no canto superior direito, depois em "Baixar fonte"; renomeie o arquivo baixado para terminar em `.tgz` se ele ainda não existir, e você deve encontrar a fonte lá, incluindo as equações para copiar e colar.

# %%
(m*m).sum().sqrt()


# %% [markdown] {"lang": "en"}
# #### Elementwise matmul

# %% [markdown] {"lang": "pt"}
# #### Elementwise matmul

# %%
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            # Any trailing ",:" can be removed
            c[i,j] = (a[i,:] * b[:,j]).sum()
    return c


# %%
# %timeit -n 10 _=matmul(m1, m2)

# %%
#export
def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)


# %%
test_near(t1,matmul(m1, m2))

# %% [markdown]
# ### Broadcasting

# %% [markdown]
# The term **broadcasting** describes how arrays with different shapes are treated during arithmetic operations.  The term broadcasting was first used by Numpy.
#
# From the [Numpy Documentation](https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html):
#
#     The term broadcasting describes how numpy treats arrays with 
#     different shapes during arithmetic operations. Subject to certain 
#     constraints, the smaller array is “broadcast” across the larger 
#     array so that they have compatible shapes. Broadcasting provides a 
#     means of vectorizing array operations so that looping occurs in C
#     instead of Python. It does this without making needless copies of 
#     data and usually leads to efficient algorithm implementations.
#     
# In addition to the efficiency of broadcasting, it allows developers to write less code, which typically leads to fewer errors.
#
# *This section was adapted from [Chapter 4](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#4.-Compressed-Sensing-of-CT-Scans-with-Robust-Regression) of the fast.ai [Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra) course.*

# %% [markdown]
# #### Broadcasting with a scalar

# %%
a

# %%
a > 0

# %% [markdown]
# How are we able to do a > 0?  0 is being **broadcast** to have the same dimensions as a.
#
# Remember above when we normalized our dataset by subtracting the mean (a scalar) from the entire data set (a matrix) and dividing by the standard deviation (another scalar)?  We were using broadcasting!
#
# Other examples of broadcasting with a scalar:

# %%
a + 1

# %%
m

# %%
2*m

# %% [markdown]
# #### Broadcasting a vector to a matrix

# %% [markdown]
# We can also broadcast a vector to a matrix:

# %%
c = tensor([10.,20,30]); c

# %%
m

# %%
m.shape,c.shape

# %%
m + c

# %%
c + m

# %% [markdown]
# We don't really copy the rows, but it looks as if we did. In fact, the rows are given a *stride* of 0.

# %%
t = c.expand_as(m)

# %%
t

# %%
m + t

# %%
t.storage()

# %%
t.stride(), t.shape

# %% [markdown]
# You can index with the special value [None] or use `unsqueeze()` to convert a 1-dimensional array into a 2-dimensional array (although one of those dimensions has value 1).

# %%
c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape
