# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] lang="pt"
# ## Multiplicação de matrizes

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

# +
import operator

def test(a,b,comparador,cname=None):
    if cname is None: 
        cname=comparador.__name__
    assert comparador(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq,'==')


# -

# ## Obter os Dados

# +
#export
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
# -

path = datasets.download_data(MNIST_URL, ext='.gz'); path

with gzip.open(path, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

type(x_train)

# Converter de numpy array para tensor

x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
rows,cols = x_train.shape
x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max()

mpl.rcParams['image.cmap'] = 'gray'

img = x_train[0]

img.shape

# Converter o tensor coluna 

img.view(28,28).type()

plt.imshow(img.view((28,28)));

# ## Initial python model

weights = torch.randn(784,10)/math.sqrt(784)

bias = torch.zeros(10)


# ### Multiplicação de Matrizes
# Definição básica

def matmul(a,b):
    arows, acols = a.shape # n_rows * n_cols
    brows, bcols = b.shape
    assert acols==brows
    c = torch.zeros(arows, bcols)
    for i in range(arows):
        for j in range(bcols):
            for k in range(acols): # or br
                c[i,j] += a[i,k] * b[k,j]
    return c


m1 = x_valid[:5]
m2 = weights

m1.shape, m2.shape


# %time t1=matmul(m1, m2)

# Eliminando o índice `k`, usando a multiplicação elemento a elemento e o método `sum` do Tensor.

def matmul(a,b):
    arows, acols = a.shape # n_rows * n_cols
    brows, bcols = b.shape
    assert acols==brows
    c = torch.zeros(arows, bcols)
    for i in range(arows):
        for j in range(bcols):
            c[i,j] = (a[i,:] * b[:,j]).sum()
    return c


# %time t1=matmul(m1, m2)

#export
def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)


test_near(t1,matmul(m1, m2))

# ### Broadcasting

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

# #### Broadcasting with a scalar

a

a > 0

# How are we able to do a > 0?  0 is being **broadcast** to have the same dimensions as a.
#
# Remember above when we normalized our dataset by subtracting the mean (a scalar) from the entire data set (a matrix) and dividing by the standard deviation (another scalar)?  We were using broadcasting!
#
# Other examples of broadcasting with a scalar:

a + 1

m

2*m

# #### Broadcasting a vector to a matrix

# We can also broadcast a vector to a matrix:

c = tensor([10.,20,30]); c

m

m.shape,c.shape

m + c

c + m

# We don't really copy the rows, but it looks as if we did. In fact, the rows are given a *stride* of 0.

t = c.expand_as(m)

t

m + t

t.storage()

t.stride(), t.shape

# You can index with the special value [None] or use `unsqueeze()` to convert a 1-dimensional array into a 2-dimensional array (although one of those dimensions has value 1).

c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape
