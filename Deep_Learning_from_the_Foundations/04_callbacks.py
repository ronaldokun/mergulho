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

# %%
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

# %%
#export
from exp.nb_03 import *
from dataclasses import dataclass

# %% [markdown]
# ## DataBunch/Learner

# %%
x_train, y_train, x_valid, y_valid = get_data()
train_dataset = Dataset(x_train, y_train)
valid_dataset = Dataset(x_valid, y_valid)
n_hidden, batch_size = 50,64
loss_func = F.cross_entropy


# %%
#export
class DataBunch():
    def __init__(self, train_dataloader, valid_dataloader):
        self.train_dataloader, self.valid_dataloader = train_dataloader, valid_dataloader
        self.c = self.train_dataset.y.max().item()+1

    @property
    def train_dataset(self): return self.train_dataloader.dataset

    @property
    def valid_dataset(self): return self.valid_dataloader.dataset


# %%
data = DataBunch(*get_dataloaders(train_dataset, valid_dataset, batch_size))


# %%
#export
def get_model(data, lr=0.5, n_hidden=50):
    m = data.train_dataset.x.shape[1]
    model = nn.Sequential(nn.Linear(m,n_hidden), nn.ReLU(), nn.Linear(n_hidden,data.c))
    return model, optim.SGD(model.parameters(), lr=lr)

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data


# %%
learn = Learner(*get_model(data), loss_func, data)


# %% [markdown]
# Factor out the connected pieces of info out of the fit() argument list
#
# `fit(epochs, model, loss_func, opt, train_dataloader, valid_dataloader)`

# %%
def fit(epochs, learn):
    for epoch in range(epochs):
        learn.model.train()
        for xb,yb in learn.data.train_dataloader:
            loss = learn.loss_func(learn.model(xb), yb)
            loss.backward()
            learn.opt.step()
            learn.opt.zero_grad()

        learn.model.eval()
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in learn.data.valid_dataloader:
                pred = learn.model(xb)
                tot_loss += learn.loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(valid_dataloader)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv


# %%
loss,acc = fit(1, learn)


# %% [markdown]
# ## CallbackHandler

# %% [markdown]
# Add callbacks so we can remove complexity from loop, and make it flexible:

# %%
def one_batch(xb, yb, cb):
    if not cb.begin_batch(xb,yb): return
    loss = cb.learn.loss_func(cb.learn.model(xb), yb)
    if not cb.after_loss(loss): return
    loss.backward()
    if cb.after_backward(): cb.learn.opt.step()
    if cb.after_step(): cb.learn.opt.zero_grad()


# %%
def all_batches(dl, cb):
    for xb,yb in dl:
        one_batch(xb, yb, cb)
        if cb.do_stop(): return


# %%
def fit(epochs, learn, cb):
    if not cb.begin_fit(learn): return
    for epoch in range(epochs):
        if not cb.begin_epoch(epoch): continue
        all_batches(learn.data.train_dataloader, cb)

        if cb.begin_validate():
            with torch.no_grad(): all_batches(learn.data.valid_dataloader, cb)
        if not cb.after_epoch(): break
    cb.after_fit()


# %%
class CallbackHandler():
    def __init__(self): self.stop,self.callbacks = False,[]

    def begin_fit(self, learn):
        self.learn,self.in_train = learn,True
        return True
    def after_fit(self): pass

    def begin_epoch(self, epoch):
        learn.model.train()
        self.in_train=True
        return True
    def begin_validate(self):
        self.learn.model.eval()
        self.in_train=False
        return True
    def after_epoch(self): return True

    def begin_batch(self, xb, yb): return True
    def after_loss(self, loss): return self.in_train
    def after_backward(self): return True
    def after_step(self): return True

    def do_stop(self):
        try:     return self.stop
        finally: self.stop = False


# %%
fit(1, learn, cb=CallbackHandler())


# %% [markdown]
# This is roughly how fastai does it now (except that the handler can also change and return `xb`, `yb`, and `loss`). But let's see if we can make things simpler and more flexible, so that a single class has access to everything and can change anything at any time. The fact that we're passing `cb` to so many functions is a strong hint they should all be in the same class!

# %% [markdown]
# ## Runner

# %%
#export
class Callback():
    _order=0
    def __init__(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)

class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.n_epochs=0.
        self.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.n_epochs += 1./self.iters
        self.n_iter   += 1

    def begin_epoch(self):
        self.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False


# %%
#export
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, tuple): return list(o)
    return [o]


# %%
#export
class Runner():
    def __init__(self, callbacks=None):
        self.stop,self.callbacks = False,[TrainEvalCallback(self)]+listify(callbacks)

    @property
    def opt(self):       return self.learn.opt
    @property
    def model(self):     return self.learn.model
    @property
    def loss_func(self): return self.learn.loss_func
    @property
    def data(self):      return self.learn.data

    def one_batch(self, xb, yb):
        self.xb,self.yb = xb,yb
        if self('begin_batch'): return
        self.pred = self.model(self.xb)
        if self('after_pred'): return
        self.loss = self.loss_func(self.pred, self.yb)
        if self('after_loss') or not self.in_train: return
        self.loss.backward()
        if self('after_backward'): return
        self.opt.step()
        if self('after_step'): return
        self.opt.zero_grad()

    def all_batches(self, dl):
        self.iters = len(dl)
        for xb,yb in dl:
            if self.stop: break
            self.one_batch(xb, yb)
            self('after_batch')
        self.stop=False

    def fit(self, epochs, learn):
        self.epochs,self.learn = epochs,learn

        try:
            if self('begin_fit'): return
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dataloader)

                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches(self.data.valid_dataloader)
                if self('after_epoch'): break

        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        for cb in sorted(self.callbacks, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True
        return False


# %%
#export
class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train

    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

class AvgStatsCallback(Callback):
    def __init__(self, run, metrics):
        super().__init__(run)
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)

    def stats(self): return self.train_stats if self.in_train else self.valid_stats

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        with torch.no_grad(): self.stats().accumulate(self.run)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


# %%
learn = Learner(*get_model(data), loss_func, data)

# %%
run = Runner()
stats = AvgStatsCallback(run, [accuracy])
run.callbacks.append(stats)

# %%
run.fit(3, learn)

# %%
loss,acc = stats.valid_stats.avg_stats
assert acc>0.9

# %% [markdown]
# ## Export

# %%
!./notebook2script.py 04_callbacks.ipynb

# %%
