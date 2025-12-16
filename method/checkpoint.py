import torch

def load_checkpoint(cfg):

    from .model import GaussianModel, Activations, Optimizer
    ch = torch.load(cfg["checkpoint"])
    if cfg["checkpoint_type"] == 'gsrt':
        print("Scene extent:", ch['scene_extent'])
        return GaussianModel(cfg,
                xyz=ch['xyz'],
                scaling=ch['scaling'],
                rotation=ch['rotation'],
                opacity=ch['opacity'],
                features_dc=ch['f_dc'],
                features_rest=ch['f_rest'],
                active_sh_degree=ch['sh_deg'],
                max_sh_degree=ch['max_sh_deg'],
                scene_extent=ch['scene_extent'],
                iteration=ch.get('iteration',0),
                optimizer=Optimizer(cfg, state_dict=ch.get('optimizer', None)),
                white_bg=ch.get("white_bg", False))

    if cfg["checkpoint_type"] == 'gsrt_old':
        ch = torch.load(cfg.checkpoint)
        return GaussianModel(cfg,
                xyz = ch['xyz'],
                scaling = ch['scaling'],
                rotation = ch['rotation'],
                opacity = ch['opacity'],
                features_dc = ch['f_dc'],
                features_rest = ch['f_rest'],
                active_sh_degree = ch['sh_deg'],
                max_sh_degree = 3,
                scene_extent = 5.2,
                white_bg = False,
                iteration = 30000
                )

    if cfg["checkpoint_type"] == '3dgs':
        act = Activations(cfg)
        ch = ch[0]
        ch = {'xyz':ch[1],'scaling':ch[4],'rotation':ch[5],'opacity':ch[6],'f_dc':ch[2],'f_rest':ch[3],'sh_deg':ch[0]}
        return GaussianModel(cfg,
                activations=act,
                xyz=ch['xyz'].detach().cuda(),
                scaling=act.scaling_inverse(act.scaling(ch['scaling'].detach().cuda())*1.5),
                rotation=ch['rotation'].detach().cuda(),
                opacity=ch['opacity'].detach().cuda(),
                features_dc=ch['f_dc'].detach().cuda(),
                features_rest=ch['f_rest'].detach().cuda(),
                active_sh_degree=ch['sh_deg'],
                scene_extent=5.2,
                max_sh_degree=3,
                white_bg=cfg.get("white_bg", False))

    if cfg["checkpoint_type"] == '3dgrt':
        N = ch['positions'].shape[0]
        return GaussianModel(cfg,
                xyz=ch['positions'].detach(),
                scaling=ch['scale'].detach(),
                rotation=ch['rotation'].detach(),
                opacity_pre_act=ch['density'].detach(),
                features_dc=ch['features_albedo'].detach().reshape(N,1,3),
                features_rest=ch['features_specular'].detach().reshape(N,15,3),
                max_sh_degree=ch['max_n_features'],
                white_bg=cfg.get("white_bg", False))

    raise ValueError(f"Unknown checkpoint type: {cfg.checkpoint_type}")

