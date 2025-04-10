from functools import partial
import numpy as np

from jax import numpy as jnp, tree, debug, lax
import jax_cosmo as jc
from jax_cosmo import Cosmology
from flbench.utils import ch2rshape, safe_div

# from jaxpm.pm import pm_forces
import jax_cosmo as jc
from jaxpm.growth import _growth_factor_ODE
from jaxpm.kernels import longrange_kernel
from jaxpm.painting import cic_paint, cic_read


def rfftk(shape):
    """
    Return wavevectors in cell units for rfftn.
    """
    kx = np.fft.fftfreq(shape[0]) * 2 * np.pi
    ky = np.fft.fftfreq(shape[1]) * 2 * np.pi
    kz = np.fft.rfftfreq(shape[2]) * 2 * np.pi

    kx = kx.reshape([-1, 1, 1])
    ky = ky.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])

    return kx, ky, kz


def invlaplace_kernel(kvec, fd=False):
    """
    Compute the inverse Laplace kernel.

    Parameters
    -----------
    kvec: list
        List of wavevectors
    fd: bool
        Finite difference kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    if fd:
        kk = sum((ki * np.sinc(ki / (2 * np.pi)))**2 for ki in kvec)
    else:
        kk = sum(ki**2 for ki in kvec)
    return - safe_div(1, kk)


def gradient_kernel(kvec, direction:int, fd=False):
    """
    Compute the gradient kernel in the given direction
    
    Parameters
    -----------
    kvec: list
        List of wavevectors
    direction: int
        Index of the direction in which to take the gradient
    fd: bool
        Finite difference kernel

    Returns
    --------
    weights: array
        Complex kernel values
    """
    ki = kvec[direction]
    if fd:
        ki = (8. * np.sin(ki) - np.sin(2. * ki)) / 6.
    return 1j * ki


def paint_kernel(kvec, order:int=2):
    """
    Compute painting kernel of given order.

    Parameters
    ----------
    kvec: list
        List of wavevectors
    order: int
        order of the kernel
        * 0: Dirac
        * 1: Nearest Grid Point (NGP)
        * 2: Cloud-In-Cell (CIC)
        * 3: Triangular-Shape Cloud (TSC)
        * 4: Piecewise-Cubic Spline (PCS)

        cf. [List and Hahn, 2024](https://arxiv.org/abs/2309.10865)

    Returns
    -------
    weights: array
        Complex kernel values
    """
    wts = [np.sinc(kvec[i] / (2 * np.pi)) for i in range(3)]
    wts = (wts[0] * wts[1] * wts[2])**order
    return wts


def pm_forces(pos, mesh_shape, mesh=None, grad_fd=False, lap_fd=False, r_split=0):
    """
    Compute gravitational forces on particles using a PM scheme
    """
    if mesh is None:
        delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), pos))
    # elif jnp.isrealobj(mesh):
    #     delta_k = jnp.fft.rfftn(mesh)
    else:
        delta_k = mesh

    # Compute gravitational potential
    kvec = rfftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec, lap_fd) * longrange_kernel(kvec, r_split=r_split)

    # # If painted field, double deconvolution to account for both painting and reading 
    # if mesh is None:
    #     print("deconv")
    #     pot_k /= paint_kernel(kvec, order=2)**2

    # Compute gravitational forces
    return jnp.stack([cic_read(jnp.fft.irfftn(- gradient_kernel(kvec, i, grad_fd) * pot_k), pos) 
                      for i in range(3)], axis=-1)


def pm_forces2(delta_k, pos, mesh_shape, lap_fd=False, grad_fd=False):
    """
    Return 2LPT source term.
    """
    kvec = rfftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec, lap_fd)

    delta2 = 0
    shear_acc = 0
    for i in range(3):
        # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
        shear_ii = gradient_kernel(kvec, i, grad_fd)**2
        shear_ii = jnp.fft.irfftn(shear_ii * pot_k)
        delta2 += shear_ii * shear_acc 
        shear_acc += shear_ii

        for j in range(i+1, 3):
            # Substract squared strict-up-triangle terms
            hess_ij = gradient_kernel(kvec, i, grad_fd) * gradient_kernel(kvec, j, grad_fd)
            delta2 -= jnp.fft.irfftn(hess_ij * pot_k)**2

    force2 = pm_forces(pos, mesh_shape, mesh=jnp.fft.rfftn(delta2), grad_fd=grad_fd, lap_fd=lap_fd)
    return force2


def lpt(cosmo:Cosmology, init_mesh, pos, a, order=2, grad_fd=False, lap_fd=False):
    """
    Compute first and second order LPT displacement, 
    e.g. Eq 3.5 and 3.7 [List and Hahn](https://arxiv.org/abs/2409.19049)
    or Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    # if jnp.isrealobj(init_mesh):
    #     delta_k = jnp.fft.rfftn(init_mesh)
    #     mesh_shape = init_mesh.shape
    # else:
    delta_k = init_mesh
    mesh_shape = ch2rshape(init_mesh.shape)

    force1 = pm_forces(pos, mesh_shape, mesh=delta_k, grad_fd=grad_fd, lap_fd=lap_fd)
    dpos = a2g(cosmo, a) * force1
    vel = force1

    if order == 2:
        force2 = pm_forces2(delta_k, pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)
        dpos -= a2gg(cosmo, a) * force2
        vel  -= a2dggdg(cosmo, a) * force2

    return dpos, vel




###########
# Growths #
###########
log10_amin: int = -3
steps: int = 128

# Growth from scale factor
def a2g(cosmo, a):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(a, cache["a"], cache["g"])

def a2gg(cosmo, a):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.growth_factor"]
    # NOTE: g2 is normalized such that gg = -3/7 * g2 ~ -3/7 * g^2
    return jnp.interp(a, cache["a"], cache["g2"]) * -3/7

def a2f(cosmo, a):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(a, cache["a"], cache["f"])

def a2ff(cosmo, a):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(a, cache["a"], cache["f2"])

def a2dggdg(cosmo, a):
    g, gg, f, ff = a2g(cosmo, a), a2gg(cosmo, a), a2f(cosmo, a), a2ff(cosmo, a)
    return safe_div(gg * ff, g * f) # NOTE: dggdg(0) = 0


# Growth from growth factor
def g2a(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["a"])

def g2gg(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.growth_factor"]
    # NOTE: g2 is normalized such that gg = -3/7 * g2 ~ -3/7 * g^2
    return jnp.interp(g, cache["g"], cache["g2"]) * -3/7

def g2f(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["f"])

def g2ff(cosmo, g):
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0), log10_amin=log10_amin, steps=steps)
    cache = cosmo._workspace["background.growth_factor"]
    return jnp.interp(g, cache["g"], cache["f2"])

def g2dggdg(cosmo, g):
    gg, f, ff = g2gg(cosmo, g), g2f(cosmo, g), g2ff(cosmo, g)
    return safe_div(gg * ff, g * f) # NOTE: dggdg(0) = 0



###########
# Solvers #
###########
def bullfrog_vf(cosmo:Cosmology, dg, mesh_shape, grad_fd=False, lap_fd=False):
    """
    BullFrog vector field.
    """
    def alpha_bf(cosmo, g0, dg):
        '''
        BullFrog growth-time integrator coefficient.
        
        See Eq. 2.3 in [List and Hahn, 2024](https://arxiv.org/abs/2106.00461)
        '''
        g1 = g0 + dg / 2
        g2 = g0 + dg

        dggdg0, dggdg2 = g2dggdg(cosmo, g0), g2dggdg(cosmo, g2)
        lin_ratio = (g2gg(cosmo, g0) + dggdg0 * dg / 2) / g1 - g1
        # NOTE: linearization of ratio (gg - g^2)/g aroung g0, evaluated at g1
        return (dggdg2 - lin_ratio) / (dggdg0 - lin_ratio)
    
    def alpha_fpm(cosmo, g0, dg):
        '''
        FastPM growth-time integrator coefficient.

        See Eq. 3.16 in [List and Hahn, 2024](https://arxiv.org/abs/2106.00461)
        '''
        g2 = g0 + dg
        a0, a2 = g2a(cosmo, g0), g2a(cosmo, g2)
        coeff0 = jc.background.Esqr(cosmo, a0)**.5 * g0 * g2f(cosmo, g0) * a0**2
        coeff2 = jc.background.Esqr(cosmo, a2)**.5 * g2 * g2f(cosmo, g2) * a2**2
        return coeff0 / coeff2

    def kick(state, g0, cosmo, dg):
        pos, vel = state
        g1 = g0 + dg / 2
        forces = pm_forces(pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)
        alpha = alpha_bf(cosmo, g0, dg)
        return pos, alpha * vel + (1 - alpha) * forces / g1
        # return pos, vel + (1 - alpha) * (forces / g1 - vel) # equivalent
        # return pos, vel + dg * forces

    def drift(state, dg):
        pos, vel = state
        return pos + vel * dg, vel
    
    def vector_field(g0, state, args):
        old = state
        state = drift(state, dg / 2)
        state = kick(state, g0, cosmo, dg)
        state = drift(state, dg / 2)
        return tree.map(lambda new, old: (new - old) / dg, state, old)
    
    # def step(state, g0):
    #     state = drift(state, dg / 2)
    #     state = kick(state, g0, cosmo, dg)
    #     state = drift(state, dg / 2)
    #     return state, None
    
    return vector_field
    # return step


from diffrax import diffeqsolve, ODETerm, SaveAt, Euler
def nbody_bf(cosmo:Cosmology, init_mesh, pos, a, n_steps=5,
              grad_fd=False, lap_fd=False, snapshots:int|list=None):
    """
    N-body simulation with BullFrog solver.
    """
    n_steps = int(n_steps)
    g = a2g(cosmo, a)
    dg = g / n_steps
    
    mesh_shape = ch2rshape(init_mesh.shape)
    terms = ODETerm(bullfrog_vf(cosmo, dg, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd))
    solver = Euler()

    vel = pm_forces(pos, mesh_shape, mesh=init_mesh, grad_fd=grad_fd, lap_fd=lap_fd)
    state = pos, vel

    if snapshots is None or (isinstance(snapshots, int) and snapshots < 2): 
        saveat = SaveAt(t1=True)
    elif isinstance(snapshots, int): 
        saveat = SaveAt(ts=a2g(cosmo, jnp.linspace(0, a, snapshots)))  
    else: 
        saveat = SaveAt(ts=a2g(cosmo, jnp.asarray(snapshots)))   

    sol = diffeqsolve(terms, solver, 0., g, dt0=dg, y0=state, max_steps=n_steps, saveat=saveat) # cosmo as args may leak
    states = sol.ys
    # debug.print("bullfrog n_steps: {n}", n=sol.stats['num_steps'])
    return states


def nbody_bf_scan(cosmo:Cosmology, init_mesh, pos, a, n_steps=5,
              grad_fd=False, lap_fd=False, snapshots:int|list=None):
    """
    No-diffrax version of N-body simulation with BullFrog solver. 
    Simpler but does not optimize for memory usage with binomial checkpointing.
    """
    g = a2g(cosmo, a)
    dg = g / n_steps
    gs = jnp.arange(n_steps) * dg

    mesh_shape = ch2rshape(init_mesh.shape)
    # vector_field = bullfrog_vf(cosmo, dg, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)

    # def step(state, g0):
    #     vf = vector_field(g0, state, None)
    #     state = tree.map(lambda x, y: x + dg * y, state, vf)
    #     return state, None
    
    step = bullfrog_vf(cosmo, dg, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd)
    
    vel = pm_forces(pos, mesh_shape, mesh=init_mesh, grad_fd=grad_fd, lap_fd=lap_fd)
    state = pos, vel

    state, _ = lax.scan(step, state, gs)
    return tree.map(lambda x: x[None], state)

















def lpt_fpm(cosmo:Cosmology, init_mesh, pos, a, order=1, grad_fd=True, lap_fd=False):
    """
    Computes first and second order LPT displacement, e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    a = jnp.atleast_1d(a)
    E = jc.background.Esqr(cosmo, a)**.5
    if jnp.isrealobj(init_mesh):
        delta_k = jnp.fft.rfftn(init_mesh)
        mesh_shape = init_mesh.shape
    else:
        delta_k = init_mesh
        mesh_shape = ch2rshape(init_mesh.shape)

    init_force = pm_forces(pos, mesh_shape, mesh=delta_k, grad_fd=grad_fd, lap_fd=lap_fd)
    dq = a2g(cosmo, a) * init_force
    p = a**2 * a2f(cosmo, a) * E * dq

    if order == 2:
        kvec = rfftk(mesh_shape)
        pot_k = delta_k * invlaplace_kernel(kvec, lap_fd)

        delta2 = 0
        shear_acc = 0
        for i in range(3):
            # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
            shear_ii = gradient_kernel(kvec, i, grad_fd)**2
            shear_ii = jnp.fft.irfftn(shear_ii * pot_k)
            delta2 += shear_ii * shear_acc 
            shear_acc += shear_ii

            for j in range(i+1, 3):
                # Substract squared strict-up-triangle terms
                hess_ij = gradient_kernel(kvec, i, grad_fd) * gradient_kernel(kvec, j, grad_fd)
                delta2 -= jnp.fft.irfftn(hess_ij * pot_k)**2

        init_force2 = pm_forces(pos, mesh_shape, mesh=jnp.fft.rfftn(delta2), grad_fd=grad_fd, lap_fd=lap_fd)
        dq2 = a2gg(cosmo, a) * init_force2 # D2 is renormalized: - D2 = 3/7 * growth_factor_second
        p2 = (a**2 * a2ff(cosmo, a) * E) * dq2

        dq -= dq2
        p  -= p2

    return dq, p


def diffrax_vf(cosmo:Cosmology, mesh_shape, grad_fd=True, lap_fd=False):
    """
    N-body ODE vector field for diffrax, e.g. Tsit5 or Dopri5

    vector field signature is (a, state, args) -> dstate, where state is a tuple (position, velocities)
    """
    def vector_field(a, state, args):
        pos, vel = state
        forces = pm_forces(pos, mesh_shape, grad_fd=grad_fd, lap_fd=lap_fd) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces
        return dpos, dvel
    return vector_field


def jax_ode_vf(cosmo:Cosmology, mesh_shape, grad_fd=True, lap_fd=False):
    """
    Return N-body ODE vector field for jax.experimental.ode.odeint

    vector field signature is (state, a, *args) -> dstate, where state is a tuple (position, velocities)
    """
    vf = diffrax_vf(cosmo, mesh_shape, grad_fd, lap_fd)
    def vector_field(state, a, *args):
        return vf(a, state, args)
    return vector_field



from diffrax import diffeqsolve, ODETerm, SaveAt, Euler, Heun, Dopri5, Tsit5, PIDController, ConstantStepSize
def nbody_tsit5(cosmo:Cosmology, mesh_shape, particles, a_lpt, a_obs, tol=1e-2,
           grad_fd=True, lap_fd=False, snapshots:int|list=None):
    if a_lpt == a_obs:
        return tree.map(lambda x: x[None], particles)
    else:
        terms = ODETerm(diffrax_vf(cosmo, mesh_shape, grad_fd, lap_fd))
        solver = Tsit5() # Tsit5 usually better than Dopri5
        controller = PIDController(rtol=tol, atol=tol, pcoeff=0.4, icoeff=1, dcoeff=0)

        if snapshots is None or (isinstance(snapshots, int) and snapshots < 2): 
            saveat = SaveAt(t1=True)
        elif isinstance(snapshots, int): 
            saveat = SaveAt(ts=jnp.linspace(a_lpt, a_obs, snapshots))   
        else: 
            saveat = SaveAt(ts=jnp.asarray(snapshots))   

        sol = diffeqsolve(terms, solver, a_lpt, a_obs, dt0=None, y0=particles,
                                stepsize_controller=controller, max_steps=1000, saveat=saveat)
        # NOTE: if max_steps > 50 for dopri5/tsit5, just quit :')
        particles = sol.ys
        debug.print("tsit5 n_steps: {n}", n=sol.stats['num_steps'])
        return particles

