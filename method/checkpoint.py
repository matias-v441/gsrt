import torch

def load_checkpoint(cfg):

    from .model import GaussianModel, Activations
    ch = torch.load(cfg["checkpoint"])
    if cfg["checkpoint_type"] == 'gsrt':
        return GaussianModel(activations=Activations(cfg),
                xyz=ch['xyz'].detach().cuda(),
                scaling=ch['scaling'].detach().cuda(),
                rotation=ch['rotation'].detach().cuda(),
                opacity=ch['opacity'].detach().cuda(),
                features_dc=ch['f_dc'].detach().cuda(),
                features_rest=ch['f_rest'].detach().cuda(),
                active_sh_degree=ch['sh_deg'],
                max_sh_degree=3,
                scene_extent=5.2)  ## TODO: store scene extent in checkpoint

    if cfg["checkpoint_type"] == '3dgs':
        act = Activations(cfg)
        return GaussianModel(activations=act,
                xyz=ch['xyz'].detach().cuda(),
                scaling=act.scaling_inverse(act.scaling(ch['scaling'].detach().cuda())*1.5),
                rotation=ch['rotation'].detach().cuda(),
                opacity=ch['opacity'].detach().cuda(),
                features_dc=ch['f_dc'].detach().cuda(),
                features_rest=ch['f_rest'].detach().cuda(),
                active_sh_degree=ch['sh_deg'],
                scene_extent=5.2, ## TODO: store scene extent in checkpoint
                max_sh_degree=3)

    if cfg["checkpoint_type"] == '3dgrt':
        N = ch['positions'].shape[0]
        return GaussianModel(cfg,
                xyz=ch['positions'].detach(),
                scaling=ch['scale'].detach(),
                rotation=ch['rotation'].detach(),
                opacity_pre_act=ch['density'].detach(),
                features_dc=ch['features_albedo'].detach().reshape(N,1,3),
                features_rest=ch['features_specular'].detach().reshape(N,15,3),
                max_sh_degree=ch['max_n_features'])
    
    raise ValueError(f"Unknown checkpoint type: {cfg.checkpoint_type}")

