from __future__ import annotations # for Union typing | in python<3.10

import pickle
import yaml

from functools import partial, wraps
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad, tree, lax

from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from numpyro.distributions import Distribution, constraints, TruncatedNormal, Uniform



def vlim(a, level=1., scale=1., axis=0):
    """
    Return robust inferior and superior limit values of an array,
    i.e. discard quantiles bilateraly on some level, and scale the margins.
    """
    vmin, vmax = jnp.quantile(a, (1 - level) / 2, axis=axis), jnp.quantile(a, (1 + level) / 2, axis=axis)
    vmean, vdiff = (vmax + vmin) / 2, scale*(vmax - vmin) / 2
    return jnp.stack((vmean - vdiff, vmean + vdiff), axis=-1)


def get_jit(*args, **kwargs):
    """
    Return custom jit function that preserves function name and documentation.
    !!! example
        ```python
            @get_jit(static_argnums=(0))
            def my_func(x,y):
                return x+y
        ```
    """
    def custom_jit(fun):
        return wraps(fun)(jit(fun, *args, **kwargs))
    return custom_jit


def nvmap(fun, n):
    """
    Nest vmap n times.
    """
    for _ in range(n):
        fun = vmap(fun)
    return fun


def safe_div(x, y):
    """
    Safe division, where division by zero is zero.
    Uses the "double-where" trick for safe gradient, 
    see https://github.com/jax-ml/jax/issues/5039
    """
    y_nozeros = jnp.where(y==0, 1, y)
    return jnp.where(y==0, 0, x / y_nozeros)



#################
# Dump and Load #
#################
class Path(type(Path()), Path):
    """Pathlib path but with right-concatenation operator. Please tell me why it is not natively implemented."""
    # See pathlib inheritance https://stackoverflow.com/questions/61689391/error-with-simple-subclassing-of-pathlib-path-no-flavour-attribute
    def __add__(self, other):
        if isinstance(other, (str, Path)):
            return Path(str(self) + str(other))
        return NotImplemented

def pdump(obj, path):
    """Pickle dump"""
    with open(path, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def pload(path):
    """Pickle load"""
    with open(path, 'rb') as file:
        return pickle.load(file)    

def ydump(obj, path):
    """YAML dump"""
    with open(path, 'w') as file:
        yaml.dump(obj, file)

def yload(path):
    """YAML load"""
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.Loader)

def numpy_array_representer(dumper, data):
    return dumper.represent_list(data.tolist())

def numpy_array_constructor(loader, node):
    return np.array(loader.construct_sequence(node))

yaml.add_representer(np.ndarray, numpy_array_representer, Dumper=yaml.SafeDumper)
yaml.add_constructor(np.ndarray, numpy_array_constructor, Loader=yaml.SafeLoader)

def ysafe_dump(obj, path):
    """YAML safe dump"""
    with open(path, 'w') as file:
        yaml.safe_dump(obj, file)

def ysafe_load(path):
    """YAML safe load"""
    with open(path, 'r') as file:
        return yaml.safe_load(file)


######################################
# Truncated Normal reparametrization #
######################################
def lowtail(x, low=-jnp.inf, high=None):
    temp = 1/6.2842226/2 # best temperature at 12 sigma
    energy = - jnp.stack(jnp.broadcast_arrays(x, low), axis=0)
    return temp * logsumexp( - energy / temp, axis=0)

def hightail(x, low=None, high=jnp.inf):
    temp = 1/6.2842226/2 # best temperature at 12 sigma
    energy = jnp.stack(jnp.broadcast_arrays(x, high), axis=0)
    return - temp * logsumexp( - energy / temp, axis=0)

def lowbody(x, low=-jnp.inf, high=jnp.inf):
    cdf_low, cdf_high = norm.cdf(low), norm.cdf(high)
    cdf_y = cdf_low + (cdf_high - cdf_low) * norm.cdf(x)
    return norm.ppf(cdf_y)

def highbody(x, low=-jnp.inf, high=jnp.inf):
    cdf_nlow, cdf_nhigh = norm.cdf(-low), norm.cdf(-high) # cdf(-x) = 1-cdf(x), more stable
    cdf_ny = cdf_nhigh - (cdf_nhigh - cdf_nlow) * norm.cdf(-x)
    return - norm.ppf(cdf_ny)

def body(x, low=-jnp.inf, high=jnp.inf):
    condlist = [x < 0.]
    funclist = [lowbody, highbody]
    return jnp.piecewise(x, condlist, funclist, low=low, high=high)    

def std2trunc(x, loc=0., scale=1., low=-jnp.inf, high=jnp.inf):
    """
    Transport standard normal variable to a general truncated normal variable. 
    """
    if scale==0:
        return loc * jnp.ones_like(x)
    low, high = (low - loc) / scale, (high - loc) / scale
    lim = 12 # switch to a more stable approx at 12 sigma, for float32
    condlist = [(x < -lim) & (low < -lim), (lim < x) & (lim < high)]
    funclist = [lowtail, hightail, body]
    return loc + scale * jnp.piecewise(x, condlist, funclist, low=low, high=high)

 


def invlowbody(y, low=-jnp.inf, high=jnp.inf):
    cdf_low, cdf_high = norm.cdf(low), norm.cdf(high)
    cdf_x = (norm.cdf(y) - cdf_low) / (cdf_high - cdf_low)
    return norm.ppf(cdf_x)

def invhighbody(y, low=-jnp.inf, high=jnp.inf):
    cdf_nlow, cdf_nhigh = norm.cdf(-low), norm.cdf(-high) # cdf(-x) = 1-cdf(x), more stable
    cdf_nx = (cdf_nhigh - norm.cdf(-y)) / (cdf_nhigh - cdf_nlow)
    return - norm.ppf(cdf_nx)

def invbody(y, low=-jnp.inf, high=jnp.inf):
    condlist = [y < 0.]
    funclist = [invlowbody, invhighbody]
    return jnp.piecewise(y, condlist, funclist, low=low, high=high)   

def invhightail(y, low=None, high=jnp.inf):
    temp = 1/6.2842226/2 # best temperature at 12 sigma
    energy, b = jnp.split(jnp.stack(jnp.broadcast_arrays(y, high, 1, -1), axis=0), 2)
    return - temp * logsumexp( - energy / temp, axis=0, b=b)

def invlowtail(y, low=-jnp.inf, high=None):
    temp = 1/6.2842226/2 # best temperature at 12 sigma
    energy, b = jnp.split(jnp.stack(jnp.broadcast_arrays(-y, -low, 1, -1), axis=0), 2)
    return temp * logsumexp( - energy / temp, axis=0, b=b)

def trunc2std(y, loc=0., scale=1., low=-jnp.inf, high=jnp.inf):
    """
    Transport a general truncated normal variable to a standard normal variable.
    """
    y, low, high = (y - loc) / scale, (low - loc) / scale, (high - loc) / scale
    lim = 12 # switch to a more stable approx at 12 sigma, for float32
    condlist = [(y < -lim) & (low < -lim), (lim < y) & (lim < high)]
    funclist = [invlowtail, invhightail, invbody]
    return jnp.piecewise(y, condlist, funclist, low=low, high=high)


class DetruncTruncNorm(Distribution):
    """
    Detruncated Truncated Normal distribution.
    Detruncation is such that a truncated normal with fiducial parameters is transformed into a standard normal.

    This means `std2trunc(DetruncTruncNorm(loc, scale, low, high, loc_fid, scale_fid), loc_fid, scale_fid, low, high)` 
    is distributed as `TruncNorm(loc, scale, low, high)`.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 
                       'low': constraints.real, 'high': constraints.real,
                       'loc_fid': constraints.real, 'scale_fid': constraints.positive, 
                       }
    support = constraints.real
    def __init__(self, loc=0., scale=1., 
                 low=-jnp.inf, high=jnp.inf, 
                 loc_fid:float=None, scale_fid:float=None, 
                 *, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
  
        self.loc_fid = loc if loc_fid is None else loc_fid
        self.scale_fid = scale if scale_fid is None else scale_fid

        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale), jnp.shape(low), jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        trunc = TruncatedNormal(self.loc, self.scale, low=self.low, high=self.high).sample(key, sample_shape)
        return trunc2std(trunc, self.loc_fid, self.scale_fid, self.low, self.high)

    def log_prob(self, value):
        fn = partial(std2trunc, loc=self.loc_fid, scale=self.scale_fid, low=self.low, high=self.high)
        log_abs_det_jac = lambda x: jnp.log(jnp.abs(grad(fn)(x)))
        # log_abs_det_jac = lambda x: analyt_log_abs_det_jac(x, self.loc_fid, self.scale_fid, self.low, self.high)
        log_pdf = TruncatedNormal(self.loc, self.scale, low=self.low, high=self.high).log_prob
        return log_pdf(fn(value)) + log_abs_det_jac(value)

class DetruncUnif(Distribution):
    """
    Detruncated Uniform distribution.
    Detruncation is such that a truncated normal with fiducial parameters is transformed into a standard normal.

    This means `std2trunc(DetruncUnif(low, high, loc_fid, scale_fid), loc_fid, scale_fid, low, high)` 
    is distributed as `Unif(low, high)`.
    """
    arg_constraints = {'low': constraints.real, 'high': constraints.real,
                       'loc_fid': constraints.real, 'scale_fid': constraints.positive, 
                       }
    support = constraints.real
    def __init__(self, low=0., high=1., 
                 loc_fid:float=None, scale_fid:float=None, 
                 *, validate_args=None):
        self.low = low
        self.high = high
        
        self.loc_fid = (high + low) / 2 if loc_fid is None else loc_fid
        self.scale_fid = (high - low) / 12**.5 if scale_fid is None else scale_fid

        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        trunc = Uniform(self.low, self.high).sample(key, sample_shape)
        return trunc2std(trunc, self.loc_fid, self.scale_fid, self.low, self.high)

    def log_prob(self, value):
        fn = partial(std2trunc, loc=self.loc_fid, scale=self.scale_fid, low=self.low, high=self.high)
        log_abs_det_jac = lambda x: jnp.log(jnp.abs(grad(fn)(x)))
        # log_abs_det_jac = lambda x: analyt_log_abs_det_jac(x, self.loc_fid, self.scale_fid, self.low, self.high)
        log_pdf = Uniform(self.low, self.high).log_prob
        return log_pdf(fn(value)) + log_abs_det_jac(value)

def analyt_log_abs_det_jac(x, loc, scale, low, high):
    # NOTE: this analytical logabsdetjac for std2trunc fails after 12sigma for float32
    low, high = (low - loc) / scale, (high - loc) / scale
    cdf_low, cdf_high = norm.cdf(low), norm.cdf(high)
    return jnp.log(scale * (cdf_high - cdf_low) * norm.pdf(x) / norm.pdf(std2trunc(x, 0., 1., low, high)))
    # cdf_y = cdf_low + (cdf_high - cdf_low) * norm.cdf(x)
    # return jnp.log(scale * (cdf_high - cdf_low) * norm.pdf(x) / norm.pdf(norm.ppf(cdf_y)))


#############################
# Fourier reparametrization #
#############################
def _rg2cgh(mesh, part="real", norm="backward"):
    """
    Return the real and imaginary parts of a complex Gaussian Hermitian tensor
    obtained by permuting and reweighting a real Gaussian tensor (3D).
    Handle the Hermitian symmetry, specifically at border faces, edges, and vertices.
    """
    shape = np.array(mesh.shape)
    assert np.all(shape % 2 == 0), "dimension lengths must be even."
    
    hx, hy, hz = shape // 2
    meshk = jnp.zeros(r2chshape(shape))

    if part == "imag":
        slix, sliy, sliz = slice(hx + 1, None), slice(hy + 1, None), slice(hz + 1, None)
    else:
        assert part == "real", "part must be either 'real' or 'imag'."
        slix, sliy, sliz = slice(1, hx), slice(1, hy), slice(1, hz)
    meshk = meshk.at[:, :, 1:-1].set(mesh[:, :, sliz])
        
    for k in [0, hz]:  # two faces
        meshk = meshk.at[:, 1:hy, k].set(mesh[:, sliy, k])
        meshk = meshk.at[1:, hy + 1:, k].set(mesh[1:, sliy, k][::-1, ::-1])
        meshk = meshk.at[0, hy + 1:, k].set(mesh[0, sliy, k][::-1])  # handle the border
        if part == "imag" and norm != "amp":
            meshk = meshk.at[:, hy + 1:, k].multiply(-1.)

        for j in [0, hy]:  # two edges per face
            meshk = meshk.at[1:hx, j, k].set(mesh[slix, j, k])
            meshk = meshk.at[hx + 1:, j, k].set(mesh[slix, j, k][::-1])
            if part == "imag" and norm != "amp":
                meshk = meshk.at[hx + 1:, j, k].multiply(-1.)

            for i in [0, hx]:  # two points per edge
                if part == "real":
                    meshk = meshk.at[i, j, k].set(mesh[i, j, k])
                    if norm != "amp":
                        meshk = meshk.at[i, j, k].multiply(2**.5)

    shape = shape.astype(float)
    if norm == "backward":
        meshk /= (2 / shape.prod())**.5
    elif norm == "forward":
        meshk /= (2 * shape.prod())**.5
    elif norm == "ortho":
        meshk /= 2**.5
    else:
        assert norm == "amp", "norm must be either 'backward', 'forward', 'ortho', or 'amp'."

    return meshk


def _cgh2rg(meshk, part="real", norm="backward"):
    """
    Return the "real" and "imaginary" partitions of a real Gaussian tensor (3D)
    obtained by permuting and reweighting the real or imaginary parts of a 
    complex Gaussian Hermitian tensor.
    Handle the Hermitian symmetry, specifically at border faces, edges, and vertices.
    """
    shape = np.array(ch2rshape(meshk.shape))
    assert np.all(shape % 2 == 0), "dimension lengths must be even."

    hx, hy, hz = shape // 2
    mesh = jnp.zeros(shape)

    if part == "imag":
        slix, sliy, sliz = slice(hx + 1, None), slice(hy + 1, None), slice(hz + 1, None)
    else:
        assert part == "real", "part must be either 'real' or 'imag'."
        slix, sliy, sliz = slice(1, hx), slice(1, hy), slice(1, hz)
    mesh = mesh.at[:, :, sliz].set(meshk[:, :, 1:-1])
        
    for k in [0, hz]:  # two faces
        mesh = mesh.at[:, sliy, k].set(meshk[:, 1:hy, k])
        mesh = mesh.at[1:, sliy, k].set(meshk[1:, hy + 1:, k][::-1, ::-1])
        mesh = mesh.at[0, sliy, k].set(meshk[0, hy + 1:, k][::-1])  # handle the border
        if part == "imag" and norm != "amp":
            mesh = mesh.at[:, sliy, k].multiply(-1.)

        for j in [0, hy]:  # two edges per face
            mesh = mesh.at[slix, j, k].set(meshk[1:hx, j, k])
            mesh = mesh.at[slix, j, k].set(meshk[hx + 1:, j, k][::-1])
            if part == "imag" and norm != "amp":
                mesh = mesh.at[slix, j, k].multiply(-1.)

            for i in [0, hx]:  # two points per edge
                if part == "real":
                    mesh = mesh.at[i, j, k].set(meshk[i, j, k])
                    if norm != "amp":
                        mesh = mesh.at[i, j, k].divide(2**.5)

    shape = shape.astype(float)
    if norm == "backward":
        mesh *= (2 / shape.prod())**.5
    elif norm == "forward":
        mesh *= (2 * shape.prod())**.5
    elif norm == "ortho":
        mesh *= 2**.5
    else:
        assert norm == "amp", "norm must be either 'backward', 'forward', 'ortho', or 'amp'."

    return mesh


def rg2cgh(mesh, norm="backward"):
    """
    Permute and reweight a real Gaussian tensor (3D) into a complex Gaussian Hermitian tensor.
    The output would therefore be distributed as the real Fourier transform of a Gaussian tensor.

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, "amp")`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    real = _rg2cgh(mesh, part="real", norm=norm)
    if norm == "amp":
        return real
    else:
        imag = _rg2cgh(mesh, part="imag", norm=norm)
        return real + 1j * imag


def cgh2rg(meshk, norm="backward"):
    """
    Permute and reweight a complex Gaussian Hermitian tensor into a real Gaussian tensor (3D).

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, "amp")`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    real = _cgh2rg(meshk.real, part="real", norm=norm)
    if norm == "amp":
        # Give same amplitude to wavevector real and imaginary part
        imag = _cgh2rg(meshk.real, part="imag", norm=norm)
    else:
        imag = _cgh2rg(meshk.imag, part="imag", norm=norm)
    return real + imag


def ch2rshape(kshape):
    """
    Complex Hermitian shape to real shape.
    
    Assume last real shape is even to lift the ambiguity.
    """
    return (*kshape[:2], 2*(kshape[2]-1))

def r2chshape(shape):
    """
    Real shape to complex Hermitian shape.
    """
    return (*shape[:2], shape[2]//2+1)






