from functools import partial
import numpy as np

from jax import numpy as jnp, tree, debug
import jax_cosmo as jc
from jax_cosmo import Cosmology
from jaxpm.painting import cic_read

from flbench.utils import std2trunc, trunc2std, rg2cgh, cgh2rg, ch2rshape, r2chshape, safe_div
from flbench.nbody import rfftk, invlaplace_kernel, a2g, a2f


# [Planck2015 XIII](https://arxiv.org/abs/1502.01589) Table 4 final column (best fit)
Planck15 = partial(Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,)

# [Planck 2018 VI](https://arxiv.org/abs/1807.06209) Table 2 final column (best fit)
Planck18 = partial(Cosmology,
    # Omega_m = 0.3111
    Omega_c=0.2607,
    Omega_b=0.0490,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,)


def lin_power_interp(cosmo=Cosmology, a=1., n_interp=256):
    """
    Return a light emulation of the linear matter power spectrum.
    """
    ks = jnp.logspace(-4, 1, n_interp)
    logpows = jnp.log(jc.power.linear_matter_power(cosmo, ks, a=a))
    # Interpolate in semilogy space with logspaced k values, correctly handles k==0,
    # as interpolation in loglog space can produce nan gradients
    pow_fn = lambda x: jnp.exp(jnp.interp(x.reshape(-1), ks, logpows, left=-jnp.inf, right=-jnp.inf)).reshape(x.shape)
    # pows = jc.power.linear_matter_power(cosmo, ks, a=a)
    # pow_fn = lambda x: jnp.interp(x.reshape(-1), ks, pows, left=0., right=0.).reshape(x.shape)
    return pow_fn


def lin_power_mesh(cosmo:Cosmology, mesh_shape, box_shape, a=1., n_interp=256):
    """
    Return linear matter power spectrum field.
    """
    pow_fn = lin_power_interp(cosmo, a=a, n_interp=n_interp)
    kvec = rfftk(mesh_shape)
    kmesh = sum((ki  * (m / l))**2 for ki, m, l in zip(kvec, mesh_shape, box_shape))**0.5
    return pow_fn(kmesh) * (mesh_shape / box_shape).prod() # from [Mpc/h]^3 to cell units
    

def kaiser_posterior(delta_obs, cosmo:Cosmology, bE, a, box_shape, gxy_count, los=None):
    """
    Return posterior mean and std fields of the linear matter field (at a=1) given the observed field,
    by assuming Kaiser model. All fields are in fourier space.
    """
    # Compute linear matter power spectrum
    mesh_shape = ch2rshape(delta_obs.shape)
    pmeshk = lin_power_mesh(cosmo, mesh_shape, box_shape)
    boost = kaiser_boost(cosmo, a, bE, mesh_shape, los)

    stds = (pmeshk / (1 + gxy_count * boost**2 * pmeshk))**.5
    # Also: stds = jnp.where(pmeshk==0., 0., pmeshk / (1 + gxy_count * evolve**2 * pmeshk))**.5
    means = stds**2 * gxy_count * boost * delta_obs
    return means, stds


def get_cosmology(**cosmo) -> Cosmology:
    """
    Return full cosmology object from cosmological params.
    """
    return Planck18(Omega_c=cosmo['Omega_m'] - Planck18.keywords['Omega_b'], 
                    sigma8=cosmo['sigma8'])


def samp2base(params:dict, config, inv=False, temp=1.) -> dict:
    """
    Transform sample params into base params.
    """
    out = {}
    for in_name, value in params.items():
        name = in_name if inv else in_name[:-1]
        out_name = in_name+'_' if inv else in_name[:-1]

        conf = config[name]
        low, high = conf.get('low', -jnp.inf), conf.get('high', jnp.inf)
        loc_fid, scale_fid = conf['loc_fid'], conf['scale_fid']
        scale_fid *= temp**.5

        # Reparametrize
        if not inv:
            if low != -jnp.inf or high != jnp.inf:
                push = lambda x: std2trunc(x, loc_fid, scale_fid, low, high)
            else:
                push = lambda x: x * scale_fid + loc_fid
        else:
            if low != -jnp.inf or high != jnp.inf:
                push = lambda x: trunc2std(x, loc_fid, scale_fid, low, high)
            else:
                push = lambda x: (x - loc_fid) / scale_fid

        out[out_name] = push(value)
    return out


def samp2base_mesh(init:dict, precond=False, transfer=None, inv=False, temp=1.) -> dict:
    """
    Transform sample mesh into base mesh, i.e. initial wavevector coefficients at a=1.
    """
    assert len(init) <= 1, "init dict should only have one or zero key"
    for in_name, mesh in init.items():
        out_name = in_name+'_' if inv else in_name[:-1]
        transfer *= temp**.5

        # Reparametrize
        if not inv:
            if precond=='direct':
                # Sample in direct space
                mesh = jnp.fft.rfftn(mesh)

            elif precond in ['fourier','kaiser','kaiser_dyn']:
                # Sample in fourier space
                mesh = rg2cgh(mesh)

            mesh *= transfer # ~ CN(0, P)
        else:
            mesh = safe_div(mesh, transfer)
            
            if precond=='direct':
                mesh = jnp.fft.irfftn(mesh)

            elif precond in ['fourier','kaiser','kaiser_dyn']:
                mesh = cgh2rg(mesh)

        return {out_name:mesh}
    return {}






def lagrangian_weights(cosmo:Cosmology, a, pos, box_shape, 
                       b1, b2, bs2, bn2, init_mesh):
    """
    Return Lagrangian bias expansion weights as in [Modi+2020](http://arxiv.org/abs/1910.07097).
    .. math::
        
        w = 1 + b_1 \\delta + b_2 \\left(\\delta^2 - \\braket{\\delta^2}\\right) + b_{s^2} \\left(s^2 - \\braket{s^2}\\right) + b_{\\nabla^2} \\nabla^2 \\delta
    """    
    # Get init_mesh at observation scale factor
    init_mesh *= a2g(cosmo, a)
    # if jnp.isrealobj(init_mesh):
    #     delta = init_mesh
    #     delta_k = jnp.fft.rfftn(delta)
    # else:
    delta_k = init_mesh
    delta = jnp.fft.irfftn(delta_k)
    # Smooth field to mitigate negative weights or TODO: use gaussian lagrangian biases
    # k_nyquist = jnp.pi * jnp.min(mesh_shape / box_shape)
    # delta_k = delta_k * jnp.exp( - kk_box / k_nyquist**2)
    # delta = jnp.fft.irfftn(delta_k)

    mesh_shape = delta.shape
    kvec = rfftk(mesh_shape)
    kk_box = sum((ki  * (m / l))**2 
            for ki, m, l in zip(kvec, mesh_shape, box_shape)) # minus laplace kernel in h/Mpc physical units

    # Init weights
    weights = 1.
    
    # Apply b1, punctual term
    delta_part = cic_read(delta, pos)
    weights = weights + b1 * delta_part

    # Apply b2, punctual term
    delta2_part = delta_part**2
    weights = weights + b2 * (delta2_part - delta2_part.mean())

    # Apply bshear2, non-punctual term
    pot_k = delta_k * invlaplace_kernel(kvec)

    shear2 = 0  
    for i, ki in enumerate(kvec):
        # Add diagonal terms
        shear2 = shear2 + jnp.fft.irfftn( - ki**2 * pot_k - delta_k / 3)**2
        for kj in kvec[i+1:]:
            # Add strict-up-triangle terms (counted twice)
            shear2 = shear2 + 2 * jnp.fft.irfftn( - ki * kj * pot_k)**2

    shear2_part = cic_read(shear2, pos)
    weights = weights + bs2 * (shear2_part - shear2_part.mean())

    # Apply bnabla2, non-punctual term
    delta_nl = jnp.fft.irfftn( - kk_box * delta_k)

    delta_nl_part = cic_read(delta_nl, pos)
    weights = weights + bn2 * delta_nl_part

    return weights




def rsd(cosmo:Cosmology, a, vel, los:np.ndarray=None):
    """
    Redshift-Space Distortion (RSD) displacement from cosmology and growth-time integrator velocity.
    Computed with respect to scale factor(s) and line-of-sight(s).

    No RSD if los is None.
    """
    if los is None:
        return jnp.zeros_like(vel)
    else:
        los = np.asarray(los)
        # growth-time integrator vel := dq / dg = v / (H * g * f), so dq_rsd := v / H = vel * g * f
        # v in (Mpc/h)*(km/s/(Mpc/h)) = km/s, so dq_rsd in Mpc/h
        dpos = vel * a2g(cosmo, a) * a2f(cosmo, a)
        # Project velocity on line-of-sight(s)
        dpos = (dpos * los).sum(-1, keepdims=True) * los
        return dpos


def kaiser_boost(cosmo:Cosmology, a, bE, mesh_shape, los:np.ndarray=None):
    """
    Return Eulerian Kaiser boost including linear growth, Eulerian linear bias, and RSD.

    No RSD if los is None.
    """
    if los is None:
        return a2g(cosmo, a) * bE
    else:
        los = np.asarray(los)
        los /= np.linalg.norm(los)
        kvec = rfftk(mesh_shape)
        kmesh = sum(kk**2 for kk in kvec)**0.5 # in cell units
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
        mumesh = safe_div(mumesh, kmesh)

        return a2g(cosmo, a) * (bE + a2f(cosmo, a) * mumesh**2)


def kaiser_model(cosmo:Cosmology, a, bE, init_mesh, los:np.ndarray=None):
    """
    Kaiser model, with linear growth, Eulerian linear bias, and RSD.
    """
    mesh_shape = ch2rshape(init_mesh.shape)
    init_mesh *= kaiser_boost(cosmo, a, bE, mesh_shape, los)
    return 1 + jnp.fft.irfftn(init_mesh) #  1 + delta


