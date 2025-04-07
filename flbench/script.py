import os
from flbench.model import FieldLevelModel, default_config
from numpyro import infer

def get_save_dir(**kwargs):
    dir = os.path.expanduser("~/scratch/pickles/")
    # dir = os.path.expanduser("/lustre/fsn1/projects/rech/fvg/uvs19wt/pickles/")
    # dir = os.path.expanduser("/lustre/fswork/projects/rech/fvg/uvs19wt/workspace/pickles/") ###############
    # dir = os.path.expanduser("/pscratch/sd/h/hsimfroy/pickles/")

    dir += f"m{kwargs['mesh_shape'][0]:d}_b{kwargs['box_shape'][0]:.1f}_ao{kwargs['a_obs']:.1f}"
    dir += f"_ev{kwargs['evolution']}_lo{kwargs['lpt_order']:d}_pc{kwargs['precond']}_ob{kwargs['observable']}/"
    return dir

def from_id(id):
    args = ParseSlurmId(id)
    config = {
          'mesh_shape':3 * (args.mesh_length,),
          'box_shape':3 * (args.box_length if args.box_length is not None else 5. * args.mesh_length,), 
          'a_obs':args.a_obs,
          'lpt_order':args.lpt_order,
          'evolution':args.evolution,
          'precond':args.precond,
          'observable':args.obs,
          'nbody_steps':5,
          }

    save_dir = get_save_dir(**config)
    from copy import deepcopy
    config = deepcopy(default_config | config)
    model = FieldLevelModel(**config)
    for k, v in model.latents.items():
        if 'scale_fid' in v:
            model.latents[k]['scale_fid'] = v['scale_fid'] * (64 / args.mesh_length)**(3/2)
    
    mcmc_config = {
        'sampler':args.sampler,
        'target_accept_prob':0.65,
        'n_samples':128 if args.mesh_length < 128 else 32, ######
        'max_tree_depth':10,
        'n_runs':30 if args.mesh_length < 128 else 60,
        'n_chains':8, ######
        'mm':args.mm,
    }
    if args.rsdb==0:
        save_dir = save_dir[:-1] + "_norsdb/"
        model.loc_fid['b1'] = 0.
        model.los = None
    if args.rsdb==1:
        save_dir = save_dir[:-1] + "_nob/"
        model.loc_fid['b1'] = 0.

    save_path = save_dir 
    save_path += f"s{mcmc_config['sampler']}_nc{mcmc_config['n_chains']:d}_ns{mcmc_config['n_samples']:d}"
    if args.mm==0:
        save_path += "_nomm"

    return model, mcmc_config, save_dir, save_path

class ParseSlurmId():
    def __init__(self, id):
        self.id = str(id)
        self.id = '311' + self.id
        # self.id = self.id + '341'
        print("True id:", self.id) #####

        dic = {}
        dic['mesh_length'] = [8,16,32,64,128,256]
        dic['evolution'] = ['kaiser','lpt','nbody']
        dic['lpt_order'] = [0,1,2]
        dic['rsdb'] = [0,1,2]
        dic['precond'] = ['direct','fourier','kaiser','kaiser_dyn']

        dic['sampler'] = ['NUTS', 'HMC', 'NUTSwG', 'NUTSwG2', 'MCLMC', 'aMCLMC']
        dic['mm'] = [0,1]

        dic['box_length'] = [None]
        dic['a_obs'] = [0.5]
        dic['obs'] = ['field']
        
        for i, (k, v) in enumerate(dic.items()):
            if i < len(self.id):
                setattr(self, k, v[int(self.id[i])])
            else:
                setattr(self, k, v[0])



def get_mcmc(model, config):
    n_samples = config['n_samples']
    n_chains = config['n_chains']
    max_tree_depth = config['max_tree_depth']
    target_accept_prob = config['target_accept_prob']
    name = config['sampler']
    mm = config['mm']
    
    if name == "NUTS":
        kernel = infer.NUTS(
            model=model,
            # init_strategy=numpyro.infer.init_to_value(values=fiduc_params)
            step_size=1e-3, 
            max_tree_depth=max_tree_depth,
            target_accept_prob=target_accept_prob,
            # adapt_step_size=False,
            adapt_mass_matrix=mm,
            )
        
    elif name == "HMC":
        kernel = infer.HMC(
            model=model,
            # init_strategy=numpyro.infer.init_to_value(values=fiduc_params),
            step_size=1e-3, 
            # Heuristic mean_length_steps_NUTS / 2, compare with gaussian pi.
            trajectory_length=4.4 / 2, 
            target_accept_prob=target_accept_prob,
            # adapt_step_size=False,
            adapt_mass_matrix=mm,
            )

    mcmc = infer.MCMC(
        sampler=kernel,
        num_warmup=n_samples,
        num_samples=n_samples, # for each run
        num_chains=n_chains,
        chain_method="vectorized",
        progress_bar=True,)
    
    return mcmc



def get_init_mcmc(model, n_chains=8):
    n_samples = 128
    max_tree_depth = 10 ######
    
    kernel = infer.NUTS(
        model=model,
        # init_strategy=numpyro.infer.init_to_value(values=fiduc_params)
        step_size=1e-3, 
        max_tree_depth=max_tree_depth,
        target_accept_prob=0.65,
        # adapt_step_size=False,
        # adapt_mass_matrix=False,
        )

    mcmc = infer.MCMC(
        sampler=kernel,
        num_warmup=n_samples,
        num_samples=n_samples, # for each run
        num_chains=n_chains,
        chain_method="vectorized",
        progress_bar=True,)
    
    return mcmc


