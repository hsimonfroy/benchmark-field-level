from __future__ import annotations # for Union typing with | in python<3.10

from functools import partial
from dataclasses import dataclass, asdict
from IPython.display import display
from pprint import pformat

import numpyro.distributions as dist
from numpyro import sample, deterministic, render_model
from numpyro.handlers import seed, condition, block, trace
from numpyro.infer.util import log_density
import numpy as np

from jax import numpy as jnp, random as jr, vmap, tree, grad, debug

from jaxpm.painting import cic_paint
from jax_cosmo import Cosmology
from flbench.bricks import (samp2base, samp2base_mesh, get_cosmology, lin_power_mesh, 
                               lagrangian_weights, rsd, kaiser_boost, kaiser_model, kaiser_posterior)
from flbench.nbody import lpt, nbody_bf, nbody_bf_scan
from flbench.metrics import spectrum, powtranscoh, deconv_paint
from flbench.utils import ydump, yload

from flbench.utils import cgh2rg, rg2cgh, ch2rshape, nvmap, safe_div, DetruncTruncNorm, DetruncUnif
from flbench.mcbench import Chains



default_config={
            # Mesh and box parameters
            'mesh_shape':3 * (64,), # int
            'box_shape':3 * (320.,), # in Mpc/h
            # Evolution
            'a_obs':0.5,
            'evolution':'lpt', # kaiser, lpt, nbody
            'nbody_steps':5,
            'nbody_snapshots':None,
            'lpt_order':2,
            # Observables
            'gxy_density':1e-3, # in galaxy / (Mpc/h)^3
            'observable':'field', # 'field', TODO: 'powspec' (with poles), 'bispec'
            'los':(0.,0.,1.),
            'poles':(0,2,4),
            # Latents
            'precond':'kaiser_dyn', # direct, fourier, kaiser, kaiser_dyn
            'latents': {'Omega_m': {'group':'cosmo', 
                                    'label':'{\\Omega}_m', 
                                    'loc':0.3111, 
                                    'scale':0.5,
                                    'scale_fid':0.02,
                                    'low': 0.05, # XXX: Omega_m < Omega_b implies nan
                                    'high': 1.},
                        'sigma8': {'group':'cosmo',
                                    'label':'{\\sigma}_8',
                                    'loc':0.8102,
                                    'scale':0.5,
                                    'scale_fid':0.02,
                                    'low': 0.,
                                    'high':jnp.inf,},
                        'b1': {'group':'bias',
                                    'label':'{b}_1',
                                    'loc':1.,
                                    'scale':0.5,
                                    'scale_fid':0.04,},
                        'b2': {'group':'bias',
                                    'label':'{b}_2',
                                    'loc':0.,
                                    'scale':2.,
                                    'scale_fid':0.02,},
                        'bs2': {'group':'bias',
                                    'label':'{b}_{s^2}',
                                    'loc':0.,
                                    'scale':2.,
                                    'scale_fid':0.08,},
                        'bn2': {'group':'bias',
                                    'label':'{b}_{\\nabla^2}',
                                    'loc':0.,
                                    'scale':2.,
                                    'scale_fid':0.2,},
                        'init_mesh': {'group':'init',
                                      'label':'{\\delta}_L',},},
            }





class Model():
    ###############
    # Model calls #
    ###############
    def _model(self, *args, **kwargs):
        raise NotImplementedError

    def model(self, *args, **kwargs):
        return self._model(*args, **kwargs)
    
    def reset(self):
        self.model = self._model

    def __call__(self):
        return self.model()
    
    def reparam(self, params, inv=False):
        return params
    
    def _block_det(self, model, hide_base=True, hide_det=True):
        base_name = self.latents.keys()
        if hide_base:
            if hide_det:
                hide_fn = lambda site: site['type'] == 'deterministic'
            else:
                hide_fn = lambda site: site['type'] == 'deterministic' and site['name'] in base_name
        else:
            if hide_det:
                hide_fn = lambda site: site['type'] == 'deterministic' and site['name'] not in base_name
            else:
                hide_fn = lambda site: False
        return block(model, hide_fn=hide_fn)

    def predict(self, rng=42, samples=None, batch_ndim=0, hide_base=True, hide_det=True, hide_samp=True, frombase=False):
        """
        Run model conditioned on samples.
        * If samples is None, return a single prediction.
        * If samples is an int or tuple, return a prediction of such shape.
        * If samples is a dict, return a prediction for each sample, assuming batch_ndim batch dimensions.
        """
        if isinstance(rng, int):
            rng = jr.key(rng)

        def single_prediction(rng, sample={}):
            # Optionally reparametrize base to sample params
            if frombase:
                sample = self.reparam(sample, inv=True) 
                # NOTE: deterministic sites have no effects with handlers.condition, but do with handlers.subsitute

            # Condition then block
            model = condition(self.model, data=sample)
            if hide_samp:
                model = block(model, hide=sample.keys())
            model = self._block_det(model, hide_base=hide_base, hide_det=hide_det)

            # Trace and return values
            tr = trace(seed(model, rng_seed=rng)).get_trace()
            return {k: v['value'] for k, v in tr.items()}

        if samples is None:
            return single_prediction(rng)
        
        elif isinstance(samples, (int, tuple)):
            if isinstance(samples, int):
                samples = (samples,)
            rng = jr.split(rng, samples)
            return nvmap(single_prediction, len(samples))(rng)
        
        elif isinstance(samples, dict):
            # All item shapes should match on the first batch_ndim dimensions,
            # so take the first item shape
            shape = jnp.shape(next(iter(samples.values())))[:batch_ndim]
            rng = jr.split(rng, shape)
            return nvmap(single_prediction, len(shape))(rng, samples)
    


    ############
    # Wrappers #
    ############
    def logpdf(self, params):
        """
        A log-density function of the model. In particular, it is the log-*probability*-density function 
        with respect to the full set of variables, i.e. E[e^logpdf] = 1.

        For unnormalized log-densities in numpyro, see https://forum.pyro.ai/t/unnormalized-densities/3251/9
        """
        return log_density(self.model, (), {}, params)[0]

    def potential(self, params):
        return - self.logpdf(params)
    
    def force(self, params):
        return grad(self.logpdf)(params) # force = - grad potential = grad logpdf
    
    def trace(self, rng):
        return trace(seed(self.model, rng_seed=rng)).get_trace()
    
    def seed(self, rng):
        self.model = seed(self.model, rng_seed=rng)

    def condition(self, data={}, frombase=False):
        # Optionally reparametrize base to sample params
        if frombase:
            data = self.reparam(data, inv=True)
        self.model = condition(self.model, data=data)

    def block(self, hide_fn=None, hide=None, expose_types=None, expose=None, hide_base=True, hide_det=True):
        """
        Selectively hides parameters in the model.
        
        Precedence is given according to the order: hide_fn, hide, expose_types, expose, (hide_base, hide_det).
        Only the set of parameters with the precedence is considered.
        The default call thus hides base and other deterministic sites, for sampling purposes.
        """
        if all(x is None for x in (hide_fn, hide, expose_types, expose)):
            self.model = self._block_det(self.model, hide_base=hide_base, hide_det=hide_det)
        else:
            self.model = block(self.model, hide_fn=hide_fn, hide=hide, expose_types=expose_types, expose=expose)

    def render(self, render_dist=False, render_params=False):
        display(render_model(self.model, render_distributions=render_dist, render_params=render_params))

    def partial(self, *args, **kwargs):
        self.model = partial(self.model, *args, **kwargs)



    #################
    # Save and load #
    #################
    def save(self, path): # with yaml because not array-like
        ydump(asdict(self), path)

    @classmethod
    def load(cls, path):
        return cls(**yload(path))









@dataclass
class FieldLevelModel(Model):
    """
    Field-level cosmological model, 
    with LPT and PM displacements, Lagrangian bias, and RSD.
    The relevant variables can be traced.

    Parameters
    ----------
    mesh_shape : array_like of int
        Shape of the mesh.
    box_shape : array_like
        Shape of the box in Mpc/h. Typically such that cell lengths would be between 1 and 10 Mpc/h.
    evolution : str
        Evolution model: 'kaiser', 'lpt', 'nbody'.
    a_obs : float
        Scale factor of observations.
    nbody_steps : int
        Number of N-body steps.
        Only used for 'nbody' evolution.
    nbody_snapshots : int or list
        Number or list of N-body snapshots to save. If None, only save last.
        Only used for 'nbody' evolution.
    lpt_order : int
        Order of LPT displacement. 
        Only used for 'lpt' evolution.
    observable : str
        Observable: 'field', 'powspec'.
    gxy_density : float
        Galaxy density in galaxy / (Mpc/h)^3
    los : array_like
        Line-of-sight direction. If None, no Redshift Space Distorsion is applied.
    poles : array_like of int
        Power spectrum poles to compute.
        Only used for 'powspec' observable.
    precond : str
        Preconditioning method: 'direct', 'fourier', 'kaiser'.
    latents : dict
        Latent variables configuration.
    """
    # Mesh and box parameters
    mesh_shape:np.ndarray
    box_shape:np.ndarray
    # Evolution
    evolution:str
    a_obs:float
    nbody_steps:int
    nbody_snapshots:int|list
    lpt_order:int
    # Observable
    observable:str
    gxy_density:float
    los:tuple
    poles:tuple
    # Latents
    precond:str
    latents:dict

    def __post_init__(self):
        self.latents = self._validate_latents()
        self.groups = self._groups(base=True)
        self.groups_ = self._groups(base=False)
        self.labels = self._labels()
        self.loc_fid = self._loc_fid()

        self.mesh_shape = np.asarray(self.mesh_shape)
        # NOTE: if x32, cast mesh_shape into float32 to avoid int32 overflow when computing products
        self.box_shape = np.asarray(self.box_shape, dtype=float)
        self.cell_shape = self.box_shape / self.mesh_shape
        if self.los is not None:
            self.los = np.asarray(self.los)
            self.los = self.los / np.linalg.norm(self.los)

        self.k_funda = 2*np.pi / np.min(self.box_shape) 
        self.k_nyquist = np.pi * np.min(self.mesh_shape / self.box_shape)
        # 2*pi factors because of Fourier transform definition
        self.gxy_count = self.gxy_density * self.cell_shape.prod()

    def __str__(self):
        out = ""
        out += f"# CONFIG\n"
        out += pformat(asdict(self), width=1)
        out += "\n\n# INFOS\n"
        out += f"cell_shape:     {list(self.cell_shape)} Mpc/h\n"
        out += f"k_funda:        {self.k_funda:.5f} h/Mpc\n"
        out += f"k_nyquist:      {self.k_nyquist:.5f} h/Mpc\n"
        out += f"mean_gxy_count: {self.gxy_count:.3f} gxy/cell\n"
        return out

    def _model(self, temp_prior=1., temp_lik=1.):
        x = self.prior(temp=temp_prior)
        x = self.evolve(x)
        return self.likelihood(x, temp=temp_lik)
    




    def prior(self, temp=1.):
        """
        A prior for cosmological model. 

        Return base parameters, as reparametrization of sample parameters.
        """
        # Sample, reparametrize, and register cosmology and biases
        tup = ()
        for g in ['cosmo', 'bias']:
            dic = self._sample(self.groups[g]) # sample               
            dic = samp2base(dic, self.latents, inv=False, temp=temp) # reparametrize
            tup += ({k: deterministic(k, v) for k, v in dic.items()},) # register base params
        cosmo, bias = tup
        cosmology = get_cosmology(**cosmo)        

        # Sample, reparametrize, and register initial conditions
        init = {}
        name_ = self.groups['init'][0]+'_'

        bE = 1 + bias['b1']
        scale, transfer = self._precond_scale_and_transfer(cosmology, bE)
        init[name_] = sample(name_, dist.Normal(0., scale)) # sample
        init = samp2base_mesh(init, self.precond, transfer=transfer, inv=False, temp=temp) # reparametrize
        init = {k: deterministic(k, v) for k, v in init.items()} # register base params

        return cosmology, bias, init


    def evolve(self, params:tuple):
        cosmology, bias, init = params

        if self.evolution=='kaiser':
            # Kaiser model
            gxy_mesh = kaiser_model(cosmology, self.a_obs, bE=1+bias['b1'], **init, los=self.los)
            gxy_mesh = deterministic('gxy_mesh', gxy_mesh)
            return gxy_mesh
                    
        # Create regular grid of particles
        pos = jnp.indices(self.mesh_shape, dtype=float).reshape(3,-1).T

        # Lagrangian bias expansion weights at a_obs (but based on initial particules positions)
        lbe_weights = lagrangian_weights(cosmology, self.a_obs, pos, self.box_shape, **bias, **init)
        # TODO: gaussian lagrangian weights

        if self.evolution=='lpt':
            # LPT displacement at a_lpt
            # NOTE: lpt assumes given mesh follows linear spectral power at a=1, and then correct by growth factor for target a_lpt
            cosmology._workspace = {}  # HACK: temporary fix
            dpos, vel = lpt(cosmology, **init, pos=pos, a=self.a_obs, order=self.lpt_order, grad_fd=False, lap_fd=False)
            pos += dpos
            pos, vel = deterministic('lpt_pos', pos), vel

        elif self.evolution=='nbody':
            cosmology._workspace = {}  # HACK: temporary fix
            pos, vel = nbody_bf(cosmology, **init, pos=pos, a=self.a_obs, n_steps=self.nbody_steps, 
                                 grad_fd=False, lap_fd=False, snapshots=self.nbody_snapshots)
            part = deterministic('nbody_pos', pos), vel
            pos, vel = tree.map(lambda x: x[-1], part)

        # RSD displacement at a_obs
        pos += rsd(cosmology, self.a_obs, vel, self.los)
        pos, vel = deterministic('rsd_pos', pos), vel

        # CIC paint weighted by Lagrangian bias expansion weights
        gxy_mesh = cic_paint(jnp.zeros(self.mesh_shape), pos, lbe_weights)
        gxy_mesh = deconv_paint(gxy_mesh, order=2)
        gxy_mesh = deterministic('gxy_mesh', gxy_mesh)

        return gxy_mesh


    def likelihood(self, mesh, temp=1.):
        """
        A likelihood for cosmological model.

        Return an observed mesh sampled from a location mesh with observational variance.
        """

        if self.observable == 'field':
            # Gaussian noise
            obs_mesh = sample('obs', dist.Normal(mesh, (temp / self.gxy_count)**.5))
            return obs_mesh # NOTE: mesh is 1+delta_obs



    def reparam(self, params:dict, fourier=True, inv=False, temp=1.):
        """
        Transform sample params into base params.
        """
        # Extract groups from params
        groups = ['cosmo','bias','init']
        key = tuple([k if inv else k+'_'] for k in groups) + tuple([['*'] + ['~'+k if inv else '~'+k+'_' for k in groups]])
        params = Chains(params, self.groups | self.groups_).get(key) # use chain querying
        cosmo_, bias_, init, rest = (q.data for q in params)
        # cosmo_, bias, init = self._get_by_groups(params, ['cosmo','bias','init'], base=inv)

        # Cosmology and Biases
        cosmo = samp2base(cosmo_, self.latents, inv=inv, temp=temp)
        bias = samp2base(bias_, self.latents, inv=inv, temp=temp)

        # Initial conditions
        if len(init) > 0:
            cosmology = get_cosmology(**(cosmo_ if inv else cosmo))
            bE = 1 + (bias_['b1'] if inv else bias['b1'])
            _, transfer = self._precond_scale_and_transfer(cosmology, bE)

            if not fourier and inv:
                init = tree.map(lambda x: jnp.fft.rfftn(x), init)

            init = samp2base_mesh(init, self.precond, transfer=transfer, inv=inv, temp=temp)
            
            if not fourier and not inv:
                init = tree.map(lambda x: jnp.fft.irfftn(x), init)

        return rest | cosmo | bias | init # possibly update rest


    ###########
    # Getters #
    ###########   
    def _validate_latents(self):
        """
        Return a validated latents config.
        """
        new = {}
        for name, conf in self.latents.items():
            new[name] = conf.copy()
            loc, scale = conf.get('loc'), conf.get('scale')
            low, high = conf.get('low'), conf.get('high')
            loc_fid, scale_fid = conf.get('loc_fid'), conf.get('scale_fid')

            assert not (loc is None) ^ (scale is None),\
                f"latent '{name}' not valid: loc and scale must be both provided or both not provided"
            assert not (low is None) ^ (high is None),\
                f"latent '{name}' not valid: low and high must be both provided or both not provided"
            
            if loc is not None: # Normal or Truncated Normal prior
                if loc_fid is None:
                    new[name]['loc_fid'] = loc
                if scale_fid is None:
                    new[name]['scale_fid'] = scale
            
            elif low is not None: # Uniform prior
                assert low <= high,\
                    f"latent '{name}' not valid: low must be lower than high"
                assert low != -jnp.inf and high != jnp.inf,\
                    f"latent '{name}' not valid: low and high must be finite for uniform distribution"
                if loc_fid is None:
                    new[name]['loc_fid'] = (low + high) / 2
                if scale_fid is None:
                    new[name]['scale_fid'] = (high - low) / 12**.5
        return new


    def _sample(self, names:str|list):
        """
        Sample latent parameters from latents config.
        """
        dic = {}
        names = np.atleast_1d(names)
        for name in names:
            conf = self.latents[name]
            loc, scale = conf.get('loc', None), conf.get('scale', None)
            low, high = conf.get('low', -jnp.inf), conf.get('high', jnp.inf)
            loc_fid, scale_fid = conf['loc_fid'], conf['scale_fid']

            if loc is not None:
                if low == -jnp.inf and high == jnp.inf:
                    dic[name+'_'] = sample(name+'_', dist.Normal((loc - loc_fid) / scale_fid, scale / scale_fid))
                else:
                    dic[name+'_'] = sample(name+'_', DetruncTruncNorm(loc, scale, low, high, loc_fid, scale_fid))
            else:
                dic[name+'_'] = sample(name+'_', DetruncUnif(low, high, loc_fid, scale_fid))
        return dic            


    def _precond_scale_and_transfer(self, cosmo:Cosmology, bE):
        """
        Return scale and transfer fields for linear matter field preconditioning.
        """
        pmeshk = lin_power_mesh(cosmo, self.mesh_shape, self.box_shape)

        if self.precond in ['direct', 'fourier']:
            scale = jnp.ones(self.mesh_shape)
            transfer = pmeshk**.5

        elif self.precond=='kaiser':
            cosmo_fid, bE_fid = get_cosmology(**self.loc_fid), 1 + self.loc_fid['b1']
            boost_fid = kaiser_boost(cosmo_fid, self.a_obs, bE_fid, self.mesh_shape, self.los)
            pmeshk_fid = lin_power_mesh(cosmo_fid, self.mesh_shape, self.box_shape)

            scale = (1 + self.gxy_count * boost_fid**2 * pmeshk_fid)**.5
            transfer = pmeshk**.5 / scale
            scale = cgh2rg(scale, norm="amp")
        
        elif self.precond=='kaiser_dyn':
            boost = kaiser_boost(cosmo, self.a_obs, bE, self.mesh_shape, self.los)

            scale = (1 + self.gxy_count * boost**2 * pmeshk)**.5
            transfer = pmeshk**.5 / scale
            scale = cgh2rg(scale, norm="amp")
        
        return scale, transfer
    

    def _groups(self, base=True):
        """
        Return groups from latents config.
        """
        groups = {}
        for name, val in self.latents.items():
            group = val['group']
            group = group if base else group+'_'
            if group not in groups:
                groups[group] = []
            groups[group].append(name if base else name+'_')
        return groups


    def _labels(self):
        """
        Return labels from latents config
        """
        labs = {}
        for name, val in self.latents.items():
            lab = val['label']
            labs[name] = lab
            labs[name+'_'] = "\\tilde"+lab
        return labs
    

    def _loc_fid(self):
        """
        Return fiducial location values from latents config.
        """
        return {k:v['loc_fid'] for k,v in self.latents.items() if 'loc_fid' in v}
    

    ###########
    # Metrics #
    ###########
    def spectrum(self, mesh, mesh2=None, kedges:int|float|list=None, comp=(0, 0), poles=0):
        return spectrum(mesh, mesh2=mesh2, box_shape=self.box_shape, 
                            kedges=kedges, comp=comp, poles=poles, los=self.los)

    def powtranscoh(self, mesh0, mesh1, kedges:int|float|list=None, comp=(0, 0)):
        return powtranscoh(mesh0, mesh1, box_shape=self.box_shape, kedges=kedges, comp=comp)


    ########################
    # Chains init and load #
    ########################
    def load_runs(self, path:str, start:int, end:int, transforms=None, batch_ndim=2) -> Chains:
        return Chains.load_runs(path, start, end, transforms, 
                                groups=self.groups | self.groups_, labels=self.labels, batch_ndim=batch_ndim)

    def reparam_chains(self, chains:Chains, fourier=False, batch_ndim=2):
        chains = chains.copy()
        chains.data = nvmap(partial(self.reparam, fourier=fourier), batch_ndim)(chains.data)
        return chains
    
    def powtranscoh_chains(self, chains:Chains, mesh0, name:str='init_mesh', 
                           kedges:int|float|list=None, comp=(0, 0), batch_ndim=2) -> Chains:
        chains = chains.copy()
        fn = nvmap(lambda x: self.powtranscoh(mesh0, x, kedges=kedges, comp=comp), batch_ndim)
        chains.data['kptc'] = fn(chains.data[name])
        return chains
    
    def kaiser_post(self, rng, delta_obs, base=False, temp=1.):
        if jnp.isrealobj(delta_obs):
            delta_obs = jnp.fft.rfftn(delta_obs)

        cosmo_fid, bE_fid = get_cosmology(**self.loc_fid), 1 + self.loc_fid['b1']
        means, stds = kaiser_posterior(delta_obs, cosmo_fid, bE_fid, self.a_obs, 
                                       self.box_shape, self.gxy_count, self.los)
        post_mesh = rg2cgh(jr.normal(rng, ch2rshape(means.shape)))
        post_mesh = temp**.5 * stds * post_mesh + means 

        init_params = self.loc_fid | {'init_mesh': post_mesh}
        if base:
            return init_params
        else:
            return self.reparam(init_params, inv=True)

    

