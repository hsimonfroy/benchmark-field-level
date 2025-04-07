from __future__ import annotations # for Union typing | in python<3.10

from pickle import dump, load, HIGHEST_PROTOCOL
from functools import partial, wraps

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad, tree, lax

from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from numpyro.distributions import Distribution, constraints, TruncatedNormal, Uniform



def pdump(obj, path):
    with open(path, 'wb') as file:
        dump(obj, file, protocol=HIGHEST_PROTOCOL)

def pload(path):
    with open(path, 'rb') as file:
        return load(file)    



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
def id_cgh(shape, part="real", norm="backward"):
    """
    Return indices and weights to permute a real Gaussian tensor of given shape (3D)
    into a complex Gaussian Hermitian tensor. 
    Handle the Hermitian symmetry, specificaly at border faces, edges, and vertices.
    """
    shape = np.asarray(shape)
    sx, sy, sz = shape
    assert sx%2 == sy%2 == sz%2 == 0, "dimensions lengths must be even."
    
    hx, hy, hz = shape//2
    kshape = (sx, sy, hz+1)
    
    weights = np.ones(kshape) / 2**.5
    if norm == "backward":
        weights *= shape.prod()**.5 
    elif norm == "forward":
        weights /= shape.prod()**.5
    else:
        assert norm=="ortho", "norm must be either 'backward', 'forward', or 'ortho'."

    id = np.zeros((3, *kshape), dtype=int)
    xyz = np.indices(shape, dtype=int)

    if part == "imag":
        slix, sliy, sliz = slice(hx+1, None), slice(hy+1, None), slice(hz+1, None)
    else:
        assert part == "real", "part must be either 'real' or 'imag'."
        slix, sliy, sliz = slice(1,hx), slice(1,hy), slice(1,hz)
    id[...,1:-1] = xyz[...,sliz]
        
    for k in [0,hz]: # two faces
        id[...,1:hy,k] = xyz[...,sliy,k]
        id[...,1:,hy+1:,k] = xyz[...,1:,sliy,k][...,::-1,::-1]
        id[...,0,hy+1:,k] = xyz[...,0,sliy,k][...,::-1] # handle the border
        if part == "imag":
            weights[:,hy+1:,k] *= -1

        for j in [0,hy]: # two edges per faces
            id[...,1:hx,j,k] = xyz[...,slix,j,k]
            id[...,hx+1:,j,k] = xyz[...,slix,j,k][...,::-1]
            if part == "imag":
                weights[hx+1:,j,k] *= -1

            for i in [0,hx]: # two points per edges
                id[...,i,j,k] = xyz[...,i,j,k]
                if part == "imag":
                    weights[i,j,k] *= 0.
                else:
                    weights[i,j,k] *= 2**.5
    
    return tuple(id), weights



def rg2cgh(mesh, amp:bool=False, norm="backward"):
    """
    Permute a real Gaussian tensor (3D) into a complex Gaussian Hermitian tensor.
    The output would therefore be distributed as the real Fourier transform of a Gaussian tensor.

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, True, norm)`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    shape = mesh.shape
    id_real, w_real = id_cgh(shape, part="real", norm=norm)
    id_imag, w_imag = id_cgh(shape, part="imag", norm=norm)
    if not amp:
        return mesh[id_real] * w_real + 1j * mesh[id_imag] * w_imag
    else:
        # Average wavevector real and imaginary power and return amplitude
        return ((mesh[id_real]**2 + mesh[id_imag]**2) / 2)**.5



def cgh2rg(meshk, amp:bool=False, norm="backward"):
    """
    Permute a complex Gaussian Hermitian tensor into a real Gaussian tensor (3D).

    This means that by setting `mean, amp = cgh2rg(meank, norm), cgh2rg(ampk, True, norm)`\\
    then `rg2cgh(mean + amp * N(0,I), norm)` is distributed as `meank + ampk * rfftn(N(0,I), norm)`

    In particular `rg2cgh(N(0,I), norm)` is distributed as `rfftn(N(0,I), norm)`
    """
    shape = ch2rshape(meshk.shape)
    id_real, w_real = id_cgh(shape, part="real", norm=norm)
    id_imag, w_imag = id_cgh(shape, part="imag", norm=norm)
    
    mesh = jnp.zeros(shape)
    if not amp:
        # NOTE: w_imag can be zero, which is not safe for gradients
        mesh = mesh.at[id_imag].set(safe_div(meshk.imag, w_imag)) 
        mesh = mesh.at[id_real].set(meshk.real / w_real)
        # NOTE: real after imag to overwrite the 2^3=8 points
    else:
        # Give same amplitude to wavevector real and imaginary part
        mesh = mesh.at[id_imag].set(meshk.real)
        mesh = mesh.at[id_real].set(meshk.real)
    return mesh


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




