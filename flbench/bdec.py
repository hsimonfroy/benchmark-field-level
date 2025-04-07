import numpy as np
from jax import numpy as jnp, vmap
from jax.typing import ArrayLike
from functools import partial

###########################
# Bayesian Decision utils #
###########################

def safe_div(x, y):
    """
    Safe division, where division by zero is zero.
    """
    y_nozeros = jnp.where(y==0, 1, y)
    return jnp.where(y==0, 0, x / y_nozeros)

def vsearchsorted(a, v, side='left', sorter=None):
    return vmap(vmap(partial(jnp.searchsorted, side=side, sorter=sorter), in_axes=(0, None)), in_axes=(None, 0))(a, v)

def cumulative_trapezoid(
    y: ArrayLike,
    x: None | ArrayLike = None,
    dx: float = 1.0,
    axis: int = -1,
    initial: ArrayLike | None = None,) -> ArrayLike:
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.
    See scipy.integrate.cumulative_trapezoid and quadax implementations.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically this value should be 0. Default is None, which means no
        value at `x[0]` is returned and `res` has one element less than `y`
        along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.
    """
    y = jnp.asarray(y)
    if x is None:
        d = dx
    else:
        x = jnp.asarray(x)
        if x.ndim == 1:
            d = jnp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the same as y.")
        else:
            d = jnp.diff(x, axis=axis)
        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the same as y.")

    d = jnp.moveaxis(d, axis, 0)
    y = jnp.moveaxis(y, axis, 0)
    res = jnp.cumsum(d * (y[1:] + y[:-1]) / 2.0, axis=0)
    res = jnp.moveaxis(res, 0, axis)

    if initial is not None:
        if not jnp.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")
        shape = list(res.shape)
        shape[axis] = 1
        res = jnp.concatenate([jnp.full(shape, initial, dtype=res.dtype), res], axis=axis)

    return res


def _broadcast_weights(w, shape, axis=None):
    """
    Try to broadcast weights to shape.
    """
    if w is None:
        w = jnp.ones(shape)
    elif jnp.ndim(w) <= 1 and axis is not None:
        w = jnp.expand_dims(w, range(jnp.ndim(w), len(shape)-axis))
        w = jnp.broadcast_to(w, shape)
    else:
        w = jnp.broadcast_to(w, shape)
    return w


def quantile(x, p, axis=0, weights=None, ord=1):
    """
    Quantile function handling weights. 
    Can compute both sample quantile and density quantile, 
    by performing density integration then interpolating the computed cdf.

    If `ord==1`, perform 1st order cdf interpolation, obtained by 0th order cumulative sum.
    0th order rectangle integration or higher order integration can be biased when 
    integration step does not tend to zero, i.e. samples `x` do not cover the support. 

    If `ord==2`, perform 2nd order cdf interpolation, obtained by 1st order trapezoid integration.
    """
    p = jnp.asarray(p)
    p_shape = p.shape
    p = p.reshape(-1)

    # Try to broadcast weights to x, then reshape
    x = jnp.atleast_1d(x)
    w = _broadcast_weights(weights, x.shape, axis)
    x, w = jnp.moveaxis(x, axis, 0), jnp.moveaxis(w, axis, 0)
    n, *out_shape = x.shape
    x, w = x.reshape(n, -1), w.reshape(n, -1)

    # Compute CDF and locate quantiles
    argsort = jnp.argsort(x, 0)
    x_sort = jnp.take_along_axis(x, argsort, 0)
    w_sort = jnp.take_along_axis(w, argsort, 0)

    if ord == 1:
        cdf = jnp.cumsum(w_sort, 0)
        cdf = safe_div(cdf, cdf[-1])
        i_high = jnp.clip(vsearchsorted(cdf.T, p, side='left'), 1, n-1)

        # Linear interpolation: 
        # Solve p = cdf_low + (q_p - q_low) * (cdf_high - cdf_low) / (q_high - q_low)
        cdf_low, cdf_high = jnp.take_along_axis(cdf, i_high-1, 0), jnp.take_along_axis(cdf, i_high, 0)
        q_low, q_high = jnp.take_along_axis(x_sort, i_high-1, 0), jnp.take_along_axis(x_sort, i_high, 0)
        q_p = q_low + (p[:,None] - cdf_low) * safe_div(q_high - q_low, cdf_high - cdf_low)

    elif ord == 2:
        cdf = cumulative_trapezoid(w_sort, x_sort, axis=0, initial=0)
        w_sort = safe_div(w_sort, cdf[-1]) # the integral is no longer sum of weights so they must be renormalized
        cdf = safe_div(cdf, cdf[-1])
        i_high = jnp.clip(vsearchsorted(cdf.T, p, side='left'), 1, n-1)

        # Quadratic interpolation:
        # Solve alpha / 2 * (q_p - q_low)**2 + w_low * (q_p - q_low) + cdf_low = p
        cdf_low = jnp.take_along_axis(cdf, i_high-1, 0)
        q_low, q_high = jnp.take_along_axis(x_sort, i_high-1, 0), jnp.take_along_axis(x_sort, i_high, 0)
        w_low, w_high = jnp.take_along_axis(w_sort, i_high-1, 0), jnp.take_along_axis(w_sort, i_high, 0)

        alphas = safe_div(w_high - w_low, q_high - q_low) # XXX: careful, can still be inf if denom very small
        delta_p = p[:,None] - cdf_low
        discr = jnp.maximum(w_low**2 + 2 * alphas * delta_p, 0) # to handle numerical errors and boundaries
        q_p = q_low + jnp.where(alphas == 0, safe_div(delta_p, w_low), safe_div(-w_low + discr**.5, alphas))
    else:
        raise NotImplementedError("Only order 1 and 2 implemented.")
    q_p = jnp.clip(q_p, q_low, q_high) # to not extrapolate
    return q_p.reshape(*p_shape, *out_shape)

def argmedian(a, axis=-1):
    """
    Return the indices corresponding to median values along the given axis. 
    If axis length is even, return the highest of the two possible indices.

    Paramters
    ---------
    a : np.ndarray
        Array to compute median indices from.
    axis : int or None, optional
        Axis along which to compute median indices. 
        The default is -1 (the last axis). If None, the flattened array is used.

    Similarly to argmax and argmin, to return values from multidimensional array,
    one must do:
    ```
    index_array = argmedian(x, axis)
    val = np.take_along_axis(x, np.expand_dims(index_array, axis=axis), axis=axis).squeeze(axis=axis)
    assert val == np.median(x, axis=axis) # if x.shape[axis] is odd
    ```
    """
    k = a.shape[axis] // 2
    return np.argpartition(a, k, axis).take(k, axis)



####################
# Credible Regions #
####################
def credint(x, p=.95, axis=0, weights=None, type='small', ord=1):
    """
    Compute the p-Credible Interval (CI),
    i.e. the interval of proba `p` which is

    * the Smallest if `type=='small'` (SCI)
    * the Lowest if `type=='low'` (LCI)
    * the Median if `type=='med'` (MCI)
    * the Highest if `type=='high'` (HCI)
    """
    if type == 'small':
        if weights is None:
            return sci_noweights(x, p, axis)
        else:
            return sci(x, p, axis, weights, ord)
    else:
        return qbci(x, p, axis, weights, type, ord)




def qbci(x, p=.95, axis=0, weights=None, type='med', ord=1):
    """
    Compute the p-Quantile-Based Credible Interval (QBCI),
    i.e. the interval of proba `p` which is

    * the Lowest if `type=='low'` (LCI)
    * the Median if `type=='med'` (MCI, alias equal-tail interval)
    * the Highest if `type=='high'` (HCI)
    """
    p = jnp.asarray(p)
    if type == 'low':
        p_low = jnp.zeros_like(p)
    elif type == 'med':
        p_low = (1-p)/2
    elif type == 'high':
        p_low = 1-p

    p_high = p_low + p
    # q_low = jnp.quantile(x, p_low, axis)
    # q_high = jnp.quantile(x, p_high, axis)
    q_low = quantile(x, p_low, axis, weights, ord)
    q_high = quantile(x, p_high, axis, weights, ord)
    return jnp.stack([q_low, q_high], -1)


def qbcr(x, p=.95, weights=None, type='med', norm='inf'):
    """
    Compute the p-Quantile-Based Credible Region (QBCR), 
    i.e. the `norm`-norm spherical region of proba `p`, where its center on dimension `i` is

    * the Lowest if `type[i]=='low'` (LCR)
    * the Median if `type[i]=='med'` (MCR)
    * the Highest if `type[i]=='high'` (HCR)

    `x` is assumed to be of shape (*n_batch, n_samples, n_dim), and `type` is broadcasted to shape (n_dim,).
    
    Return both the region center and radius (in `norm`-norm). Center is of shape (*n_batch, n_dim,), and radius is of shape (*n_p, *n_batch,).
    """
    x = jnp.atleast_2d(x)
    type = np.broadcast_to(type, x.shape[-1])
    quants = quantile(x, [0., 1/2, 1.], -2, weights)
    conds = [
        type == 'low',
        type == 'med',
        type == 'high',
        ]
    center = jnp.select(conds, quants)

    dists = jnp.linalg.norm(x - center[...,None,:], ord=norm, axis=-1)
    # radius = jnp.quantile(dists, p, -1)
    radius = quantile(dists, p, -1, weights)
    return center, radius
    

def sci_noweights(x, p:float=.95, axis=0):
    """
    Compute the p-Smallest Credible Interval (SCI) / p-Highest Density Interval (HDI),
    i.e. the smallest interval of proba `p`.
    
    Non vmapable over `p` nor jitable.
    """
    x = jnp.moveaxis(x, axis, 0)
    x_sort = jnp.sort(x, axis=0)
    n = x.shape[0]
    # Round for better estimation at low number of sample, and also handle case proba near 1.
    i_length = min(int(jnp.rint(p * n)), n-1) # NOTE: this makes function non-vmapable/jitable

    intervals_low = x_sort[: (n - i_length)] # no need to consider all low bounds
    intervals_high = x_sort[i_length:]  # no need to consider all high bounds
    intervals_length = intervals_high - intervals_low # all intervals with given proba
    
    i_low = intervals_length.argmin(axis=0)
    i_high = i_low + i_length
    q_low = jnp.take_along_axis(x_sort, i_low[None], 0)[0]
    q_high = jnp.take_along_axis(x_sort, i_high[None], 0)[0]
    return jnp.stack([q_low, q_high], axis=-1)


def sci(x, p=.95, axis=0, weights=None, ord=1):
    """
    Compute the p-Smallest Credible Interval (SCI) / p-Highest Density Interval (HDI),
    i.e. the smallest interval of proba `p`.
    """
    p = jnp.asarray(p)
    p_shape = p.shape
    p = jnp.reshape(p, -1)

    # Try to broadcast weights to x, then reshape
    x = jnp.atleast_1d(x)
    w = _broadcast_weights(weights, x.shape, axis)
    x, w = jnp.moveaxis(x, axis, 0), jnp.moveaxis(w, axis, 0)
    n, *out_shape  = x.shape
    x, w = x.reshape(n, -1), w.reshape(n, -1)

    # Compute CDF
    argsort = jnp.argsort(x, 0)
    x_sort = jnp.take_along_axis(x, argsort, 0)
    w_sort = jnp.take_along_axis(w, argsort, 0)

    if ord == 1:
        cdf = jnp.cumsum(w_sort, 0)
    elif ord == 2:
        cdf = cumulative_trapezoid(w_sort, x_sort, axis=0, initial=0)
        w_sort = safe_div(w_sort, cdf[-1]) # the integral is no longer sum of weights so they must be renormalized
    else:
        raise NotImplementedError("Only order 1 and 2 implemented.")
    cdf = safe_div(cdf, cdf[-1])

    # Find all the possible low quantiles
    # Shapes n,m ; p,1,1 ; n,m ; m -> p,n,m
    q_lows = jnp.where(cdf <= (1-p)[:,None,None], x_sort, x_sort[0])

    # Get corresponding high quantiles
    if x.shape[1] > 1:
        # Shapes n,m ; p,n,m ; n,m -> p,n,m
        q_highs = vmap(lambda x, p, w: quantile(x, p, 0, w, ord), 
                       in_axes=(-1,-1,-1), out_axes=-1)(x_sort, cdf + p[:,None,None], w_sort)
    else: # no need to vmap
        # Shapes n ; p,n ; n -> p,n
        q_highs = quantile(x_sort[:,0], cdf[:,0] + p[:,None], 0, w_sort[:,0], ord)
        q_lows = q_lows[:,:,0]

    # Minimize interval
    lengths = q_highs - q_lows
    i_small = lengths.argmin(axis=1)
    q_low = jnp.take_along_axis(q_lows, i_small[:,None], 1)
    q_high = jnp.take_along_axis(q_highs, i_small[:,None], 1)
    return jnp.stack([q_low, q_high], axis=-1).reshape(*p_shape, *out_shape, 2)


def scr(x, p=.95):
    """
    Compute the p-Smallest Credible Region (SCR) / p-Highest Density Region (HDR),
    i.e. the smallest region of proba `p`.

    Return both a KDE mesh of the samples density, 
    and the density level corresponding to `p`.
    """
    kde_mesh = 'foo'
    mesh_sort = jnp.sort(kde_mesh.reshape(-1), 0)
    cum_mesh = mesh_sort[::-1].cumsum(0)
    # i_high = jnp.clip(vsearchsorted(cum_mesh.T, p, side='left'), 1, n-1)
    
    
    pass # TODO, and vectorize over p




