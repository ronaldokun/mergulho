# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

# %%
#export
from exp.nb_06 import *

# %% [markdown]
# ## ConvNet

# %%
x_train,y_train,x_valid,y_valid = get_data()

x_train,x_valid = normalize_to(x_train,x_valid)
train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)

nh,bs = 50,512
loss_func = F.cross_entropy

data = DataBunch(*get_dls(train_ds, valid_ds, bs))


# %%
def conv2d(ni, nf, ks=3, stride=2, **kwargs):
    return nn.Sequential(
        nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride), GeneralRelu(**kwargs))


# %%
#export
def get_cnn_layers(data, nfs, **kwargs):
    nfs = [1] + nfs
    return [conv2d(nfs[i], nfs[i+1], **kwargs)
            for i in range(len(nfs)-1)] + [
        nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c)]

def get_cnn_model(data, nfs, **kwargs): return nn.Sequential(*get_cnn_layers(data, nfs, **kwargs))


# %%
def mnist_resize(x): return x.view(-1, 1, 28, 28)

def append_stats(hook, mod, inp, outp):
    hook.means.append(outp.mean())
    hook.stds .append(outp.std())

def init_cnn(m):
    for l in m:
        if isinstance(l, nn.Sequential): init.kaiming_normal_(l[0].weight, a=0.1)

def get_runner(model, lr=0.6, cbs=None, loss_func = F.cross_entropy):
    opt = optim.SGD(model.parameters(), lr=lr)
    learn = Learner(model, opt, loss_func, data)
    return learn, Runner([AvgStatsCallback([accuracy]), CudaCallback(),
        BatchTransformXCallback(resize_tfm(1,28,28))] + listify(cbs))

def get_learn_run(nfs, lr, cbs=None):
    model =  nn.Sequential(*get_cnn_layers(data, nfs, leak=0.1, sub=0.4, maxv=6.))
    init_cnn(model)
    return get_runner(model, lr=lr, cbs=cbs)


# %%
nfs = [8,16,32,32]

# %%
learn,run = get_learn_run(nfs, 0.1)


# %%
# %time run.fit(1, learn)

# %% [markdown]
# ## Batchnorm

# %% [markdown]
# ### Custom

# %%
class BatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        # NB: pytorch bn mom is opposite of what you'd expect
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('vars',  torch.zeros(1,nf,1,1))
        self.register_buffer('means', torch.zeros(1,nf,1,1))
        self.steps = 0

    def update_stats(self, x):
        self.steps += 1
        m = x.mean((0,2,3), keepdim=True)
        v = x.var ((0,2,3), keepdim=True)
        self.means.detach_()
        self.vars.detach_()
        mom = 1. if self.steps==1 else self.mom
        self.means.lerp_(m, mom)
        self.vars.lerp_ (v, mom)
        
    def forward(self, x):
        if self.training: self.update_stats(x)
        x = (x-self.means).div_((self.vars+self.eps).sqrt())
        return x.mul_(self.mults).add_(self.adds)


# %%
def conv2d(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    # No bias needed if using bn
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=not bn),
              GeneralRelu(**kwargs)]
    if bn: layers.append(BatchNorm(nf))
    return nn.Sequential(*layers)


# %%
learn,run = get_learn_run(nfs, 0.8)

# %%
with Hooks(learn.model, append_stats) as hooks:
    run.fit(1, learn)
    fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
    for h in hooks:
        ax0.plot(h.means[:10])
        ax1.plot(h.stds[:10])
        h.remove()
    plt.legend(range(6));
    
    fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
    for h in hooks:
        ax0.plot(h.means)
        ax1.plot(h.stds);

# %%
learn,run = get_learn_run(nfs, 1.0)


# %%
# %time run.fit(3, learn)

# %% [markdown]
# ### Bias corrected

# %% [markdown]
# See XL file *mom_bias.xlsx*.

# %%
class BatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        # NB: pytorch bn mom is opposite of what you'd expect
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('vars',  torch.zeros(1,nf,1,1))
        self.register_buffer('means', torch.zeros(1,nf,1,1))
        self.steps = 0

    def update_stats(self, x):
        self.steps += 1
        m = x.mean((0,2,3), keepdim=True)
        v = x.var ((0,2,3), keepdim=True)
        self.means.detach_()
        self.vars.detach_()
        mom = self.mom
        self.means.lerp_(m, mom)
        self.vars.lerp_ (v, mom)
        
    def forward(self, x):
        if self.training: self.update_stats(x)
        means = self.means
        vars  = self.vars
        if self.steps<100:
            corr = 1-(1-self.mom)**self.steps
            means = means/corr
            vars  = vars /corr
        x = (x-means).div_((vars+self.eps).sqrt())
        return x.mul_(self.mults).add_(self.adds)


# %%
learn,run = get_learn_run(nfs, 1.0)

# %%
# %time run.fit(3, learn)

# %% [markdown]
# ### jit

# %%
#export
from torch.jit import ScriptModule, script_method, script
from typing import *


# %%
class BatchNorm(ScriptModule):
    __constants__ = ['mom', 'eps']
    def __init__(self, nf, mom=0.1, eps=1e-3):
        super().__init__()
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('vars',  torch.zeros(1,nf,1,1))
        self.register_buffer('means', torch.zeros(1,nf,1,1))
        self.register_buffer('steps', tensor(0))

    @script_method
    def update_stats(self, x):
        self.steps += 1
        self.means.detach_()
        self.vars.detach_()
        if bool(self.steps>100) and bool(self.steps % 2): return
        m = x.mean((0,2,3), keepdim=True)
        v = x.var ((0,2,3), keepdim=True)
        if bool(self.steps==1):
            self.means.copy_(m)
            self.vars .copy_(v)
        else:
            self.means.copy_(self.means*(1-self.mom) + m*self.mom)
            self.vars .copy_(self.vars *(1-self.mom) + v*self.mom)

    @script_method
    def forward(self, x):
        if self.training: self.update_stats(x)
        x = (x-self.means).div((self.vars+self.eps).sqrt())
        return x.mul(self.mults).add(self.adds)


# %%
learn,run = get_learn_run(nfs, 0.8)


# %%
# %time run.fit(3, learn)

# %% [markdown]
# ### Builtin batchnorm

# %%
#export
def conv2d(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=not bn),
              GeneralRelu(**kwargs)]
    if bn: layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


# %%
learn,run = get_learn_run(nfs, 1.0)

# %%
# %time run.fit(3, learn)

# %% [markdown]
# ### With scheduler

# %%
sched = combine_scheds([0.3, 0.7], [sched_lin(0.2, 1.), sched_lin(1., 0.1)]) 

# %%
learn,run = get_learn_run([8, 16, 32, 64], 1., cbs=ParamScheduler('lr', sched))

# %%
run.fit(8, learn)


# %% [markdown]
# ## More norms

# %% [markdown]
# ### Layer norm

# %% [markdown]
# From [the paper](https://arxiv.org/abs/1607.06450): "*batch normalization cannot be applied to online learning tasks or to extremely large distributed models where the minibatches have to be small*".

# %% [markdown]
# General equation for a norm layer with learnable affine:
#
# $$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$

# %%
class LayerNorm(ScriptModule):
    __constants__ = ['eps']
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mult = nn.Parameter(tensor(1.))
        self.add  = nn.Parameter(tensor(0.))

    @script_method
    def forward(self, x):
        m = x.mean((1,2,3), keepdim=True)
        v = x.std ((1,2,3), keepdim=True)
        x = (x-m) / ((v+self.eps).sqrt())
        return x*self.mult + self.add


# %%
def conv2d(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=True),
              GeneralRelu(**kwargs)]
    if bn: layers.append(LayerNorm())
    return nn.Sequential(*layers)


# %%
learn,run = get_learn_run(nfs, 0.5)


# %%
# %time run.fit(3, learn)

# %% [markdown]
# *Thought experiment*: can this distinguish green frogs from red frogs (assuming you're using it before the first conv)?

# %% [markdown]
# ### Instance norm

# %% [markdown]
# From [the paper](https://arxiv.org/abs/1607.08022): 

# %% [markdown]
# The key difference between **contrast** and batch normalization is that the latter applies the normalization to a  whole batch of images instead for single ones:
#
# \begin{equation}\label{eq:bnorm}
#     y_{tijk} =  \frac{x_{tijk} - \mu_{i}}{\sqrt{\sigma_i^2 + \epsilon}},
#     \quad
#     \mu_i = \frac{1}{HWT}\sum_{t=1}^T\sum_{l=1}^W \sum_{m=1}^H x_{tilm},
#     \quad
#     \sigma_i^2 = \frac{1}{HWT}\sum_{t=1}^T\sum_{l=1}^W \sum_{m=1}^H (x_{tilm} - mu_i)^2.
# \end{equation}
#
# In order to combine the effects of instance-specific normalization and batch normalization, we propose to replace the latter by the *instance normalization* (also known as *contrast normalization*) layer:
#
# \begin{equation}\label{eq:inorm}
#     y_{tijk} =  \frac{x_{tijk} - \mu_{ti}}{\sqrt{\sigma_{ti}^2 + \epsilon}},
#     \quad
#     \mu_{ti} = \frac{1}{HW}\sum_{l=1}^W \sum_{m=1}^H x_{tilm},
#     \quad
#     \sigma_{ti}^2 = \frac{1}{HW}\sum_{l=1}^W \sum_{m=1}^H (x_{tilm} - mu_{ti})^2.
# \end{equation}

# %%
class InstanceNorm(nn.Module):
    __constants__ = ['eps']
    def __init__(self, eps=1e-0):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        m = x.mean((2,3), keepdim=True)
        v = x.std ((2,3), keepdim=True)
        return (x-m) / ((v+self.eps).sqrt())


# %%
def conv2d(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=True),
              GeneralRelu(**kwargs)]
    if bn: layers.append(InstanceNorm())
    return nn.Sequential(*layers)


# %%
learn,run = get_learn_run(nfs, 0.1)


# %%
# %time run.fit(3, learn)

# %% [markdown]
# *Question*: why can't this classify anything?

# %% [markdown]
# ### Group norm

# %% [markdown]
# *From the PyTorch docs:*

# %% [markdown]
# `GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)`
#
# The input channels are separated into `num_groups` groups, each containing
# ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
# separately over the each group. $\gamma$ and $\beta$ are learnable
# per-channel affine transform parameter vectorss of size `num_channels` if
# `affine` is ``True``.
#
# This layer uses statistics computed from input data in both training and
# evaluation modes.
#
# Args:
# -    num_groups (int): number of groups to separate the channels into
# -    num_channels (int): number of channels expected in input
# -    eps: a value added to the denominator for numerical stability. Default: 1e-5
# -    affine: a boolean value that when set to ``True``, this module
#         has learnable per-channel affine parameters initialized to ones (for weights)
#         and zeros (for biases). Default: ``True``.
#
# Shape:
# - Input: :math:`(N, num\_channels, *)`
# - Output: :math:`(N, num\_channels, *)` (same shape as input)
#
# Examples::
#
#     >>> input = torch.randn(20, 6, 10, 10)
#     >>> # Separate 6 channels into 3 groups
#     >>> m = nn.GroupNorm(3, 6)
#     >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
#     >>> m = nn.GroupNorm(6, 6)
#     >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
#     >>> m = nn.GroupNorm(1, 6)
#     >>> # Activating the module
#     >>> output = m(input)

# %% [markdown]
# ## Fix small batch sizes

# %% [markdown]
# ### Exponentially Weighted Layer Norm

# %%
class EWLayerNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom,self.eps = mom,eps
        self.mult = nn.Parameter(tensor(1.))
        self.add  = nn.Parameter(tensor(0.))
        self.register_buffer('batch', tensor(0.))
        self.register_buffer('sum'  , tensor(0.))
        self.register_buffer('sqr'  , tensor(0.))
        self.register_buffer('count', tensor(0.))

    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sum.detach_()
        self.sqr.detach_()
        mom = 1 if self.batch==0 else self.mom
        self.sum  .add_(mom, x.sum()        - self.sum  )
        self.sqr  .add_(mom, x.pow(2).sum() - self.sqr  )
        self.count.add_(mom, x.numel()      - self.count)
        self.batch += bs

    def forward(self, x):
        if self.training: self.update_stats(x)
        mean = self.sum/self.count
        var  = self.sqr/self.count - mean.pow(2)
        x = (x-mean)/((var+self.eps).sqrt())
        return x*self.mult + self.add


# %%
def conv2d(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=True),
              GeneralRelu(**kwargs)]
    if bn: layers.append(EWLayerNorm(nf))
    return nn.Sequential(*layers)


# %%
learn,run = get_learn_run(nfs, 0.5)


# %%
# %time run.fit(3, learn)

# %% [markdown]
# ### Exponentially Weighted Batch Norm

# %%
class EWBatchNorm(nn.Module):
    __constants__ = ['mom', 'eps']
    def __init__(self, nf, mom=0.01, eps=1e-5):
        super().__init__()
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('sums',  torch.zeros(1,nf,1,1))
        self.register_buffer('sqrs',  torch.zeros(1,nf,1,1))
        self.register_buffer('batch', tensor(0.))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('step',  tensor(0.))
        self.register_buffer('dbias', tensor(0.))

    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        self.mom1 = 1 - (1-self.mom)**x.shape[0]
        s  = x.sum(dims, keepdim=True)
        ss = x.pow(2).sum(dims, keepdim=True)
        c  = x.numel()/nc
        self.sums.copy_ (self.sums .add(self.mom1, s -self.sums))
        self.sqrs.copy_ (self.sqrs .add(self.mom1, ss-self.sqrs))
        self.count.copy_(self.count.add(self.mom1, c-self.count))
        self.dbias.copy_(self.mom1 + self.dbias*(1-self.mom1))
        self.batch += bs
        self.step  += 1

    def forward(self, x):
        if self.training: self.update_stats(x)
        sums = self.sums
        sqrs = self.sqrs
        if self.step<100:
            sums = sums/self.dbias
            sqrs = sqrs/self.dbias
        means = self.sums/self.count
        vars  = self.sqrs/self.count - means.pow(2)
        if bool(self.batch < 20): vars.clamp_min_(0.01)
        x = (x-means).div((vars+self.eps).sqrt())
        return x*self.mults + self.adds


# %%
def conv2d(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=not bn),
              GeneralRelu(**kwargs)]
    if bn: layers.append(EWBatchNorm(nf))
    return nn.Sequential(*layers)


# %%
learn,run = get_learn_run(nfs, 1.0)

# %%
# %time run.fit(3, learn)

# %% [markdown]
# ## Export

# %%
!./notebook2script.py 07_batchnorm.ipynb

# %%
