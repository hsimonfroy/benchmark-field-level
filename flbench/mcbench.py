
import os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, jit

from numpyro.infer import MCMC
from numpyro.diagnostics import print_summary
from getdist import MCSamples

from flbench.utils import pdump, pload, nvmap
from flbench.metrics import multi_ess, multi_gr

from dataclasses import dataclass, fields
from collections import UserDict
from jax import tree, tree_util



@tree_util.register_pytree_node_class
# @partial(tree_util.register_dataclass, data_fields=['data'], meta_fields=['groups']) # JAX >=0.4.27
@dataclass
class Samples(UserDict):
    """
    Global slicing and indexing s[1:3,2]
    Querying with groups, e.g.
    s['abc', 'c', 'd'], s['*~abc'], s['*','~c']
    s[['abc','c'],['d']], s[['*~abc']], s[['*','~c']]
    """
    # see also https://github.com/cosmodesi/cosmoprimo/blob/33011906d323f56c32c77ba034a6679b02f16876/cosmoprimo/emulators/tools/samples.py#L43
    data: dict
    groups: dict = None # dict of list of key

    NoneOrEmpty = object() # a sentinel to return empty dict if key is not found
    def __post_init__(self):
        if isinstance(self.data, Samples):
            otherdict = self.data.asdict()
            self.data = self.data.data # avoid nested Samples
        else:
            otherdict = {}
        selfdict = {field.name: (getattr(self, field.name) or {}).copy() for field in fields(self)} # handle None and shallow copy
        for k in selfdict:
            self.__setattr__(k, otherdict.get(k, {}) | selfdict[k]) # inherit attributes if not updated


    def __getitem__(self, key, default_fn=None):
        # Global indexing and slicing
        if self._istreeof(key, (int, slice, type(Ellipsis), np.ndarray, jnp.ndarray)):
            return tree.map(lambda x: x[key], self)

        # Querying with groups
        elif self._istreeof(key, str):
            if isinstance(key, str):
                key = self._parse_key([key])
                if len(key) == 1:
                    return self._get(key[0], default_fn)
                else:
                    return tuple(self._get(k, default_fn) for k in key)

            elif isinstance(key, list): # construct new instance
                if default_fn is self.NoneOrEmpty:
                    data = {k: self.data[k] for k in self._parse_key(key) if k in self.data}
                else:
                    data = {k: self._get(k, default_fn) for k in self._parse_key(key)}
                return type(self)(**self.asdict() | {'data': data})
            
            elif isinstance(key, tuple):
                key = self._parse_key(key)
                if len(key) == 1:
                    return self.__getitem__(key[0], default_fn)
                else:
                    return tuple(self.__getitem__(k, default_fn) for k in key)
    

    def _parse_key(self, key):
        newkey = []
        for k in key:
            if isinstance(k, list):
                newkey += [k] # handle list
            elif isinstance(k, str):
                if k.startswith('*~'): # all except
                    k = k[2:]
                    g = [k] if k in self else self.data.keys() if k=='*' else self.groups.get(k, [k])
                    # NOTE: parse self first for compatibility with dict, update, etc.
                    newkey += list(self.data.keys() - set(g))
                elif k.startswith('~'): # except
                    k = k[1:]
                    g = [k] if k in self else self.data.keys() if k=='*' else self.groups.get(k, [k])
                    for kk in g:
                        newkey.remove(kk) if kk in newkey else None
                else:
                    g = [k] if k in self else self.data.keys() if k=='*' else self.groups.get(k, [k])
                    newkey += list(g)
            else:
                raise KeyError(k)
        return newkey
    
    
    def _istreeof(self, obj, type):
        return tree.all(tree.map(lambda x: isinstance(x, type), obj))
    
    def _get(self, key, default_fn=None):
        # Rewrite dict get method to raise KeyError by default.
        if key in self.data:
            return self.data[key]
        elif default_fn is None:
            raise KeyError(key)
        elif default_fn is self.NoneOrEmpty:
            # NOTE: NoneOrEmpty is a sentinel that behaves as lambda k: None 
            # except when ask for subdict where it will return empty dict for keys not found.
            return None
        else:
            return default_fn(key)

    def get(self, key, default_fn=NoneOrEmpty):
        """
        If key is not found, get will by default return None when asked for value,
        and empty dict when asked for subdict. To get subdict with None value, use `default_fn=lambda k:None`.
        """
        return self.__getitem__(key, default_fn)
     

    def asdict(self):
        # NOTE: dataclasses.asdict makes deepcopy, cf. https://github.com/python/cpython/issues/88071
        # here, attributes are only shallow copied
        return {field.name: getattr(self, field.name).copy() for field in fields(self)}

    def __copy__(self): 
        # NOTE: UserDict copy() would not copy other attributes than data
        return type(self)(**self.asdict())

    @property
    def shape(self):
        return tree.map(jnp.shape, self.data)
    
    @property
    def ndim(self):
        return tree.map(jnp.ndim, self.data)
    
    @property
    def dtype(self):
        return tree.map(jnp.dtype, self.data)
    
    @property
    def size(self):
        return tree.map(jnp.size, self.data)
    
    # NOTE: no need with register_dataclass JAX >=0.4.27
    def tree_flatten(self):
        return (self.data,), (self.groups,)
    
    @classmethod
    # NOTE: no need with register_dataclass JAX >=0.4.27
    def tree_unflatten(cls, aux, data):
        return cls(*data, *aux)
    
    
    def __or__(self, other):
        newdict = self.asdict()
        if isinstance(other, Samples):
            otherdict = other.asdict()
            for k in otherdict:
                if k in newdict:
                    newdict[k] = newdict[k] | otherdict[k]
                else:
                    return NotImplemented
        elif isinstance(other, UserDict):
            newdict |= {'data': self.data | other.data}
        elif isinstance(other, dict):
            newdict |= {'data': self.data | other}
        else:
            return NotImplemented
        return type(self)(**newdict)
    
    def __ror__(self, other):
        newdict = self.asdict()
        if isinstance(other, Samples):
            otherdict = other.asdict()
            for k in otherdict:
                if k in newdict:
                    newdict[k] = otherdict[k] | newdict[k]
                else:
                    return NotImplemented
        elif isinstance(other, UserDict):
            newdict |= {'data': other.data | self.data}
        elif isinstance(other, dict):
            newdict |= {'data': other | self.data}
        else:
            return NotImplemented
        return type(self)(**newdict)

    def __ior__(self, other): 
        # NOTE: inplace or, so dict |= UserDict remains a dict, contrary to dic | UsertDict
        if isinstance(other, Samples):
            otherdict = other.asdict()
            selfdict = self.asdict()
            for k in selfdict:
                self.__setattr__(k, selfdict[k] | otherdict.get(k, {}) )
            return self
        else:
            return super().__ior__(other)


    ##############
    # Transforms #
    ##############
    def concat(self, *others, axis=0):
        return tree.map(lambda x, *y: jnp.concatenate((x, *y), axis=axis), self, *others)

    def stackby(self, names:str|list=None, remove=True, axis=-1):
        """
        Stack variables by groups, optionally removing individual variables.

        names can be variable names (no stacking) or group names.
        """
        if names is None:
            names = self.groups
        elif isinstance(names, str):
            names = [names]

        new = self.copy()
        for k in names:
            if k not in self: # if name is a variable do noting
                if len(self.groups[k]) == 1:
                    new.data[k] = self[k]
                else:
                    new.data[k] = jnp.stack(self[k], axis=axis)

                # Remove individual variables
                if remove:
                    for k in self.groups[k]:
                        new.data.pop(k)
        return new










@tree_util.register_pytree_node_class
@dataclass
class Chains(Samples):
    labels: dict = None
        
    # NOTE: no need with register_dataclass JAX >=0.4.27
    def tree_flatten(self):
        return (self.data,), (self.groups, self.labels)


    @classmethod
    def load_runs(cls, path:str, start:int, end:int, transforms=None, groups=None, labels=None, batch_ndim=2):
        """
        Load and append runs (or extra fields) saved in different files with same name except index.

        Both runs `start` and `end` are included.
        Runs are concatenated along last batch dimension.
        """
        print(f"Loading: {os.path.basename(path)}, from run {start} to run {end} (included)")
        for i_run in range(start, end + 1):
            if not os.path.exists(path + f"_{i_run}.npz"):
                if i_run == start:
                    raise FileNotFoundError(f"File {path}_{i_run}.npz does not exist")
                else:
                    print(f"File {path}_{i_run}.npz does not exist, stopping at run {i_run-1}")
                    end = i_run - 1
                    break
            
        if transforms is None:
            transforms = []
        transforms = np.atleast_1d(transforms)
        conc_axis = max(batch_ndim-1, 0)

        @jit
        def transform(samples):
            for trans in transforms:
                samples = trans(samples)
            return samples

        for i_run in range(start, end + 1):
            # Load
            part = dict(jnp.load(path+f"_{i_run}.npz")) # better than pickle for dict of array-like
            # part = dict(jnp.load(path+f"_{i_run}.p", allow_pickle=True))
            part = cls(part, groups=groups, labels=labels)
            part = transform(part)

            # Init or append samples
            if batch_ndim == 0:
                part = tree.map(lambda x: x[None], part)

            if i_run == start:
                samples = part
            else:
                samples = samples.concat(part, axis=conc_axis)
                del part  

        return samples

    ######################
    # General Transforms #
    ######################
    def splitrans(self, transform, n, axis=1):
        """
        Apply transform on n splits along given axis.
        Stack n values along first axis.
        """
        assert n <= jnp.shape(self[next(iter(self))])[axis], "n should be less (<=) than the length of given axis."
        out = tree.map(lambda x: jnp.array_split(x, n, axis), self)
        out = transform(out)
        
        for k in out:
            out[k] = jnp.stack(out[k])
        return out

    def cumtrans(self, transform, n, axis=1):
        """
        Apply transform on n cumulative slices along given axis.
        Stack n values along first axis.
        """
        length = jnp.shape(self[next(iter(self))])[axis]
        ends = jnp.rint(jnp.arange(1,n+1) / n * length).astype(int)
        out = tree.map(lambda x: [], self)
        for end in ends:
            part = tree.map(lambda x: x[axis*(slice(None),) + (slice(None,end),)], self)
            part = transform(part)
            for k in self:
                out[k].append(part[k])

        for k in self:
            out[k] = jnp.stack(out[k])
        return out
    
    def choice(self, n, names:str|list=None, rng=42, batch_ndim=2, replace=False):
        """
        Select a random subsample of size n along given axis for variables selected by names.
        names can be variable names or group names.
        """
        if names is None:
            names = list(self)
        else:
            names = np.atleast_1d(names)

        if isinstance(rng, int):
            rng = jr.key(rng)
        fn = lambda x: jr.choice(rng, x.reshape(-1), shape=(n,), replace=replace)
        fn = nvmap(fn, batch_ndim)

        for k in names:
            self |= tree.map(fn, self.get([k]))
        return self

    def thin(self, thinning=None, moment=None, axis:int=1):
        # All item shapes should match on given axis so take the first item shape
        length = jnp.shape(next(iter(self.values())))[axis]
        if thinning is None:
            n_split = 1
        else:
            n_split = max(np.rint(length / thinning), 1)
        
        if moment is None:
            fn = lambda c: Chains.last(c, axis=axis)
        else:
            fn = lambda c: Chains.moment(c, m=moment, axis=axis)
        out = self.splitrans(fn, n_split, axis=axis)
        return tree.map(lambda x: jnp.moveaxis(x, 0, axis), out)
    
    def flatten(self, batch_ndim=2):
        """
        Flatten all non-batch dimensions, creating new keys.
        Update groups and labels accordingly.
        """
        # Flatten data
        data = {}
        labels = {}
        substitute = {}

        for k, v in self.data.items():
            shape = jnp.shape(v)[batch_ndim:]
            if len(shape) == 0:
                # Get data and labels
                data[k] = v
                if k in self.labels:
                    labels[k] = self.labels[k]
            else:
                substitute[k] = []
                for ids in product(*map(range, shape)):
                    sufx = "[{}]".format(",".join(map(str, ids)))
                    slices = batch_ndim * (slice(None),)
                    for id in ids:
                        slices += (id,)
                    
                    # Update data and labels
                    data[k + sufx] = v[slices]
                    if k in self.labels:
                        labels[k + sufx] = self.labels[k] + sufx

                    # Register substitution to update groups
                    substitute[k].append(k + sufx)
        
        # Update groups
        groups = {}
        for g, gl in self.groups.items():
            groups[g] = [] # make a new list to not overwrite shallow copied groups
            for k in gl:
                if k in substitute:
                    groups[g] += substitute[k]
                else:
                    groups[g].append(k)
        return Chains(data, groups=groups, labels=labels)


    #####################
    # Metric Transforms #
    #####################
    def metric(self, fn, *others, axis=None):
        """
        Tree map chains but treat 'n_evals' item separately by summing it along axis.
        `self` and `others` should have matching keys, except possibly 'n_evals'.
        """
        name = "n_evals"
        infos, rest = self.get(([name], ['*~'+name]))
        infos = tree.map(lambda x: jnp.sum(x, axis), infos)

        others_new = ()
        for other in others:
            others_new += (other[['*~'+name]],)
            
        return infos | tree.map(fn, rest, *others_new)
    
    def last(self, axis=1):
        return self.metric(lambda x: jnp.take(x, -1, axis), axis=axis)
    

    def moment(self, m:int|list=(0,1,2), axis=1):
        if isinstance(m, int):
            fn = lambda x: jnp.sum(x**m, axis)
        else:
            m = jnp.asarray(m)
            fn = lambda x: jnp.sum(x[...,None]**m, axis)
        return self.metric(fn, axis=axis)
    
    def center_moment(self, axis=-1):
        def center(moments, axis):
            moments = jnp.moveaxis(moments, axis, 0)
            count = moments[0]
            mean = moments[1] / count
            std = (moments[2] / count - mean**2)**.5
            return jnp.stack((mean, std), axis)
        
        return self.metric(lambda x: center(x, axis), axis=())
    
    def cmoment(self, axis=1):
        fn = lambda x: jnp.stack((x.mean(axis), x.std(axis)), -1)
        return self.metric(fn, axis=axis)

    # def mse(self, truth, axis=0):
    #     return self.metric(lambda x, y: jnp.mean((x-y)**2, axis), truth)
    
    def mse_cmoment(self, true_cmom, axis=None):
        cmom = self.cmoment(axis=1)
        true_cmom = Chains(true_cmom, self.groups, self.labels) # cast into Chains

        def mse_mom(est, true, axis):
            n_chains = est.shape[0]
            est = jnp.moveaxis(est, -1, 0)
            true = jnp.moveaxis(true, -1, 0)
            sqrerr_mean = ((est[0] - true[0]) / true[1])**2 / n_chains
            sqrerr_std = 2 * ((est[1] - true[1]) / true[1])**2 / n_chains
            # NOTE: such square errors are asymptotically N(0, 1 / n_eff)^2 = chi^2(1) / n_eff
            return jnp.stack((sqrerr_mean.mean(axis), sqrerr_std.mean(axis)))
            # NOTE: asymptotically chi^2(n_c * n_d) / (n_c * n_d * n_eff)

        return cmom.metric(lambda x, y: mse_mom(x, y, axis), true_cmom)

    def eval_times_mse(self, truth, axis=None):
        mse_mom = self.mse_cmoment(truth, axis=axis)
        name = "n_evals" 
        infos, rest = mse_mom[[name], ['*~'+name]]
        return infos | tree.map(lambda x: infos[name] * x, rest)


    def multi_ess(self, axis=None):
        return self.metric(lambda x: multi_ess(x, axis=axis))
    
    def eval_per_ess(self, axis=None):
        ess = self.multi_ess(axis=axis)
        name = "n_evals" 
        infos, rest = ess[[name], ['*~'+name]]
        return infos | tree.map(lambda x: infos[name] / x, rest)


    ############
    # Plotting #
    ############
    def to_getdist(self, label=None):
        samples, names, labels = [], [], []
        for k, v in self.data.items():
            # Flatten all chains
            samples.append(v.reshape(-1))
            names.append(k)
            labels.append(self.labels.get(k, None))
        return MCSamples(samples=samples, names=names, labels=labels, label=label)
    
    def print_summary(self, group_by_chain=True):
        print_summary(self.data, group_by_chain=group_by_chain)


    def plot(self, names:str|list=None, batch_ndim=2, grid=True, log=False):
        """
        Plot chains. names can be variable names or group names.
        """
        if names is None:
            names = list(self)
        else:
            names = list(np.atleast_1d(names))
        
        # Concatenate extra dims or expand missing dims
        n_conc = max(batch_ndim-2, 0)
        n_exp = max(2-batch_ndim, 0)
        def conc_exp_fn(v):
            for _ in range(n_conc):
                v = jnp.concatenate(v)
            return jnp.expand_dims(v, axis=range(n_exp))
        conc = tree.map(conc_exp_fn, self[names])

        # All item shapes should match on the first batch_ndim dimensions,
        # so take the first item shape
        n_chains = jnp.shape(next(iter(conc.values())))[0]

        fig = plt.gcf()
        subfigs = fig.subfigures(len(names), 1)
        subfigs = np.atleast_1d(subfigs)
        for subfig, name in zip(subfigs, names):

            subfig.suptitle(f"{name}")
            axs = subfig.subplots(1, n_chains, sharey='row')
            axs = np.atleast_1d(axs)
            subfig.subplots_adjust(wspace=0)

            for i_n, (k, v) in enumerate(conc[[name]].items()):
                for i_c, ax in enumerate(axs):
                    label = conc.labels.get(k)
                    ax.plot(v[i_c], label=k if label is None else '$'+label+'$')
                    if log: 
                        ax.set_yscale('log')
                    ax.grid(grid)

                    xlab = ax.get_xticklabels()
                    if xlab and i_n==0:
                        plt.setp(xlab[:2], visible=False)

                ax.legend()









###############
# NumPyro API #
###############

# TODO: can select var_names directly in numpyro run api

def save_run(mcmc:MCMC, i_run:int, path:str, extra_fields:list=None, group_by_chain:bool=True):
    """
    Save one run of MCMC sampling, with extra fields and last state.
    """
    # Save samples (and extra fields)
    samples = mcmc.get_samples(group_by_chain)

    if extra_fields is not None:
        extra = mcmc.get_extra_fields(group_by_chain)
        if "num_steps" in extra: # renaming num_steps into clearer n_evals
            n_evals = extra.pop("num_steps")
            samples.update(n_evals=n_evals)
        samples |= extra
        del extra

    jnp.savez(path+f"_{i_run}.npz", **samples) # better than pickle for dict of array-like
    del samples

    # Save or overwrite last state
    pdump(mcmc.last_state, path+f"_last_state.p")
    

def sample_and_save(mcmc:MCMC, path:str, start:int=0, end:int=1, extra_fields=(),
                    rng=42, group_by_chain:bool=True, init_params=None) -> MCMC:
    """
    Warmup and run MCMC, saving the specified variables and extra fields.
    If `mcmc.num_warmup > 0`, first step is a warmup step.
    So to continue a run, simply do before:
    ```
    mcmc.num_warmup = 0
    mcmc.post_warmup_state = last_state
    ```
    """
    if isinstance(rng, int):
        rng = jr.key(rng)

    # Warmup sampling
    if mcmc.num_warmup > 0:
        print(f"\nrun {start}/{end} (warmup)")

        # Warmup
        mcmc.warmup(rng, collect_warmup=True, extra_fields=extra_fields, init_params=init_params)
        save_run(mcmc, start, path, extra_fields, group_by_chain)

        # Print warmup last state infos
        print("mean_acc_prob:", mcmc.last_state.mean_accept_prob, 
            "\nstep_size:", mcmc.last_state.adapt_state.step_size, 
            "\nsqrt_invmm:", mcmc.last_state.adapt_state.mass_matrix_sqrt_inv)

        # Handling rng key and destroy init_params
        rng_run = mcmc.post_warmup_state.rng_key
        init_params = None
        start += 1
    else:
        rng_run = rng

    # Run sampling
    for i_run in range(start, end+1):
        print(f"\nrun {i_run}/{end}")
            
        # Run
        mcmc.run(rng_run, extra_fields=extra_fields, init_params=init_params)
        save_run(mcmc, i_run, path, extra_fields)

        # Init next run at last state
        mcmc.post_warmup_state = mcmc.last_state
        rng_run = mcmc.post_warmup_state.rng_key
    return mcmc



    