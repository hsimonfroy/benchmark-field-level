import numpy as np
import jax.numpy as jnp
from functools import partial

from scipy.special import legendre
from jaxpm.growth import growth_rate, growth_factor
from flbench.nbody import rfftk, paint_kernel
from flbench.utils import safe_div, ch2rshape

from numpyro.diagnostics import effective_sample_size, gelman_rubin
# from blackjax.diagnostics import effective_sample_size
from jax_cosmo import Cosmology




############
# Spectrum #
############
def _waves(mesh_shape, box_shape, kedges, los):
    """
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    box_shape : tuple of float
        Physical dimensions of the box.
    kedges : None, int, float, or list
        * If None, set dk to twice the minimum.
        * If int, specifies number of edges.
        * If float, specifies dk.
    los : array_like
        Line-of-sight vector.

    Returns
    -------
    kedges : ndarray
        Edges of the bins.
    kmesh : ndarray
        Wavenumber mesh.
    mumesh : ndarray
        Cosine mesh.
    rfftw : ndarray
        RFFT weights accounting for Hermitian symmetry.
    """
    kmax = np.pi * np.min(mesh_shape / box_shape) # = knyquist

    if isinstance(kedges, (type(None), int, float)):
        if kedges is None:
            dk = 2*np.pi / np.min(box_shape) * 2 # twice the fundamental wavenumber
        if isinstance(kedges, int):
            dk = kmax / kedges # final number of bins will be kedges-1
        elif isinstance(kedges, float):
            dk = kedges
        kedges = np.arange(0, kmax, dk) + dk/2 # from dk/2 to kmax-dk/2

    kvec = rfftk(mesh_shape) # cell units
    kvec = [ki * (m / b) for ki, m, b in zip(kvec, mesh_shape, box_shape)] # h/Mpc physical units
    kmesh = sum(ki**2 for ki in kvec)**0.5

    if los is None:
        mumesh = 0.
    else:
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
        mumesh = safe_div(mumesh, kmesh)

    rfftw = np.full_like(kmesh, 2)
    rfftw[..., 0] = 1
    if mesh_shape[-1] % 2 == 0:
        rfftw[..., -1] = 1

    return kedges, kmesh, mumesh, rfftw


def spectrum(mesh, mesh2=None, box_shape=None, kedges:int|float|list=None, 
             comp=(0, 0), poles=0, los:np.ndarray=None):
    """
    Compute the auto and cross spectrum of 3D fields, with multipole.
    """
    # Initialize
    mesh_shape = np.array(mesh.shape)
    if box_shape is None:
        box_shape = mesh_shape
    else:
        box_shape = np.asarray(box_shape)

    if los is not None:
        los = np.asarray(los)
        los /= np.linalg.norm(los)
    pls = np.atleast_1d(poles)

    # FFTs and deconvolution
    if isinstance(comp, int):
        comp = (comp, comp)

    mesh = jnp.fft.rfftn(mesh, norm='ortho')
    kvec = rfftk(mesh_shape) # cell units
    mesh /= paint_kernel(kvec, order=comp[0])

    if mesh2 is None:
        mmk = mesh.real**2 + mesh.imag**2
    else:
        mesh2 = jnp.fft.rfftn(mesh2, norm='ortho')
        mesh2 /= paint_kernel(kvec, order=comp[1])
        mmk = mesh * mesh2.conj()

    # Binning
    kedges, kmesh, mumesh, rfftw = _waves(mesh_shape, box_shape, kedges, los)
    n_bins = len(kedges) + 1
    dig = np.digitize(kmesh.reshape(-1), kedges)

    # Count wavenumber in bins
    kcount = np.bincount(dig, weights=rfftw.reshape(-1), minlength=n_bins)
    kcount = kcount[1:-1]

    # Average wavenumber values in bins
    # kavg = (kedges[1:] + kedges[:-1]) / 2
    kavg = np.bincount(dig, weights=(kmesh * rfftw).reshape(-1), minlength=n_bins)
    kavg = kavg[1:-1] / kcount

    # Average wavenumber power in bins
    pow = jnp.empty((len(pls), n_bins))
    for i_ell, ell in enumerate(pls):
        weights = (mmk * (2*ell+1) * legendre(ell)(mumesh) * rfftw).reshape(-1)
        if mesh2 is None:
            psum = jnp.bincount(dig, weights=weights, length=n_bins)
        else: 
            # NOTE: bincount is really slow with complex numbers, so bincount real and imag parts
            psum_real = jnp.bincount(dig, weights=weights.real, length=n_bins)
            psum_imag = jnp.bincount(dig, weights=weights.imag, length=n_bins)
            psum = (psum_real**2 + psum_imag**2)**.5
        pow = pow.at[i_ell].set(psum)
    pow = pow[:,1:-1] / kcount * (box_shape / mesh_shape).prod() # from cell units to [Mpc/h]^3

    # kpow = jnp.concatenate([kavg[None], pk])
    if poles==0:
        return kavg, pow[0]
    else:
        return kavg, pow



def transfer(mesh0, mesh1, box_shape, kedges:int|float|list=None, comp=(False, False)):
    if isinstance(comp, int):
        comp = (comp, comp)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow0 = pow_fn(mesh0, comp=comp[0])
    ks, pow1 = pow_fn(mesh1, comp=comp[1])
    return ks, (pow1 / pow0)**.5


def coherence(mesh0, mesh1, box_shape, kedges:int|float|list=None, comp=(False, False)):
    if isinstance(comp, int):
        comp = (comp, comp)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow01 = pow_fn(mesh0, mesh1, comp=comp)  
    ks, pow0 = pow_fn(mesh0, comp=comp[0])
    ks, pow1 = pow_fn(mesh1, comp=comp[1])
    return ks, pow01 / (pow0 * pow1)**.5


def powtranscoh(mesh0, mesh1, box_shape, kedges:int|float|list=None, comp=(False, False)):
    if isinstance(comp, int):
        comp = (comp, comp)
    pow_fn = partial(spectrum, box_shape=box_shape, kedges=kedges)
    ks, pow01 = pow_fn(mesh0, mesh1, comp=comp)  
    ks, pow0 = pow_fn(mesh0, comp=comp[0])
    ks, pow1 = pow_fn(mesh1, comp=comp[1])
    return ks, pow1, (pow1 / pow0)**.5, pow01 / (pow0 * pow1)**.5
    




def deconv_paint(mesh, order=2):
    """
    Deconvolve the mesh by the paint kernel of given order.
    """
    if jnp.isrealobj(mesh):
        kvec = rfftk(mesh.shape)
        mesh = jnp.fft.rfftn(mesh)
        mesh /= paint_kernel(kvec, order)
        mesh = jnp.fft.irfftn(mesh)
    else:
        kvec = rfftk(ch2rshape(mesh.shape))
        mesh /= paint_kernel(kvec, order)
    return mesh






def kaiser_formula(cosmo:Cosmology, a, lin_kpow, bE, poles=0):
    """
    bE is the Eulerien linear bias
    """
    poles = jnp.atleast_1d(poles)
    beta = growth_rate(cosmo, a) / bE
    k, pow = lin_kpow
    pow *= growth_factor(cosmo, a)**2

    weights = np.ones(len(poles)) * bE**2
    for i_ell, ell in enumerate(poles):
        if ell==0:
            weights[i_ell] *= (1 + beta * 2/3 + beta**2 /5)
        elif ell==2:
            weights[i_ell] *= (beta * 4/3 + beta**2 *4/7) 
        elif ell==4:
            weights[i_ell] *= beta**2 * 8/35
        else: 
            raise NotImplementedError(
                "Handle only poles of order ell=0, 2 ,4. ell={ell} not implemented.")
        
    pow = jnp.moveaxis(pow[...,None] * weights, -1, -2)
    return k, pow




#################
# Chain Metrics #
#################
def geomean(x, axis=None):
    return jnp.exp(jnp.mean(jnp.log(x), axis=axis))

def harmean(x, axis=None):
    return 1 / jnp.mean(1 / x, axis=axis)

def multi_ess(x, axis=None):
    return harmean(effective_sample_size(x), axis=axis)

def multi_gr(x, axis=None):
    """
    In the order of (1+nc/mESS)^(1/2), with nc the number of chains.
    cf. https://arxiv.org/pdf/1812.09384 and mESS := HarMean(ESS)
    """
    return jnp.mean(gelman_rubin(x)**2, axis=axis)**.5

