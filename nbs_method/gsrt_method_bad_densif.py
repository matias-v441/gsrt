from typing import Any, Dict
from nerfbaselines import Method
from nerfbaselines import Dataset, ModelInfo
from nerfbaselines import Cameras, CameraModel
from nerfbaselines import cameras
import torch

from extension import GaussiansTracer

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



class _TraceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, setup, part_opac,part_xyz,part_scale,part_rot,part_sh,part_color):
        tracer = setup['tracer']
        tracer.load_gaussians(part_xyz,part_rot,part_scale,part_opac,part_sh,setup['sh_deg'],part_color)
        out = tracer.trace_rays(setup['ray_origins'],setup['ray_directions'],
                                setup['width'],setup['height'],
                                False,torch.tensor(0))
        ctx.setup = setup
        return out["radiance"]

    @staticmethod
    def backward(ctx, *grad_outputs):
        dout_dC = grad_outputs[0]
        setup = ctx.setup
        out = setup['tracer'].trace_rays(setup['ray_origins'],setup['ray_directions'],
                                         setup['width'],setup['height'],
                                         True,dout_dC)
        grad_xyz = out["grad_xyz"].cpu()
        grad_opacity = out["grad_opacity"].cpu()[:,None]
        grad_scale = out["grad_scale"].cpu()
        grad_rot = out["grad_rot"].cpu()
        grad_sh = out["grad_sh"].cpu()
        grad_color = out["grad_color"].cpu()
        for grad,n in [(grad_xyz,"grad_xyz"),
                       (grad_opacity,"grad_opacity"),
                       (grad_scale,"grad_scale"),
                       (grad_rot,"grad_rot"),
                       (grad_sh,"grad_sh"),
                       (grad_color,"grad_color")]:
            nan_mask = torch.isnan(grad)
            if torch.any(nan_mask):
                print(f"found NaN grad in {n}")
            grad[nan_mask] = 0.
        return None,grad_opacity,grad_xyz,grad_scale,grad_rot,grad_sh,grad_color

        
def trace_function(setup,*args):
    out = _TraceFunction.apply(setup,*args)
    L = torch.mean(torch.sum((out-setup['target_img'])**2,dim=1))
    return L,out


class GSRTMethod(Method):
    def __init__(self, *,
                checkpoint: str = None, 
                train_dataset: Dataset = None,
                config_overrides: Dict[str, Any] = None):
        super().__init__()

        self.train_dataset = train_dataset
        self.hparams = {
            "init_num_points": 100000,
            "learning_rate": .001,
            # densification
            "gradient_threshold": 0.05,
            "opacity_threshold": 0.01,
            "split_scale": 1.6,
            "densify": True
        }
        self.opacity_threshold = self.hparams['opacity_threshold']
        self.gradient_threshold = self.hparams['gradient_threshold']
        self.split_scale = self.hparams['split_scale']
        self.percent_dense = self.hparams['percent_dense']
        self.lr = self.hparams['learning_rate']
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.percent_dense = 0.01

        self.checkpoint = checkpoint

        self.best_model = None
        self.best_loss = torch.inf

        self.device = torch.device("cuda:0")
        self.tracer = GaussiansTracer(self.device)

        torch.manual_seed(0)

        if checkpoint is None:
            part_num = self.hparams['init_num_points']
            scene_min = torch.tensor([-1.,-1.,-1.])
            scene_max = torch.tensor([1.,1.,1.])
            self.scene_extent = torch.max(scene_max-scene_min)
            self.xyz = (torch.rand(part_num,3)-.5)*(scene_max-scene_min)*.5+(scene_max+scene_min)*.5
            self.scale = torch.log(torch.ones(1,3).repeat(part_num,1)*0.01)
            self.rot = torch.hstack([torch.ones(part_num,1),torch.zeros(part_num,3)])
            self.opac = torch.ones(part_num,1)*.4
            self.sh = torch.zeros(part_num,16,3).contiguous()
            self.color = torch.ones(part_num,3).contiguous()*torch.tensor([0.,1.,0.])
            self.active_sh_degree = 3
        else:
            scene_min = torch.tensor([-1.,-1.,-1.])
            scene_max = torch.tensor([1.,1.,1.])
            self.scene_extent = torch.max(scene_max-scene_min)
            self.best_model = torch.load(checkpoint)
            self.opac,self.xyz,self.scale,self.rot,self.sh,self.color = self.best_model
            self.active_sh_degree = 3
            # gaussians,it = torch.load(checkpoint)
            # self.xyz = gaussians[1].detach().cpu()
            # print("scene dim", torch.mean(self.xyz,dim=0))
            # part_num = self.xyz.shape[0]
            # self.scale = torch.exp(gaussians[4].detach().cpu())*1.5
            # self.rot = gaussians[5].detach().cpu()
            # self.opac = torch.sigmoid(gaussians[6].detach().cpu())
            # features_dc = gaussians[2].detach().cpu()
            # features_rest = gaussians[3].detach().cpu()
            # self.sh = torch.cat((features_dc,features_rest),dim=1).contiguous()
            # self.active_sh_degree = gaussians[0]


        self.params = [self.opac,self.xyz,self.scale,self.rot,self.sh,self.color]

        self.tracer.load_gaussians(self.xyz,self.rot,torch.exp(self.scale),torch.sigmoid(self.opac),
                                   self.sh,self.active_sh_degree,self.color)

        for p in self.params:
            p.requires_grad_()

        #self.sh.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.params,lr=self.lr, weight_decay=1e-4, eps=1e-2)

        self.xyz_grad_abs_acc = torch.zeros(self.params[1].shape[0])

        self.viewpoint_ids = torch.arange(1)


    def save(self, path):
        torch.save(self.best_model,f'{path}/checkpoint.pt')

    @classmethod
    def get_method_info(cls):
        return {
            # Method ID is provided by the registry
            "method_id": "",  

            # Supported camera models (e.g., pinhole, opencv, ...)
            "supported_camera_models": frozenset(("pinhole",)),

            # Features required for training (e.g., color, points3D_xyz, ...)
            "required_features": frozenset(("color",)),

            # Declare supported outputs
            "supported_outputs": ("color","transmittance","debug_map_0","debug_map_1"),
        }
    
    def get_info(self) -> ModelInfo:
        return {
            **self.get_method_info(),
            "loaded_checkpoint": self.checkpoint
        }

    @torch.no_grad()
    def prune(self):
        mask = (torch.sigmoid(self.params[0])<self.opacity_threshold).squeeze()
        if torch.sum(mask) == 0:
            return
        self.params = [p[~mask].detach() for p in self.params]
        print("PRUNE", torch.sum(mask))
        return mask

    @torch.no_grad()
    def clone(self, grad_xyz, grad_xyz_abs):
        mask_grad = grad_xyz_abs>self.gradient_threshold
        mask_scale = torch.max(torch.exp(self.params[2]),dim=1).values<self.percent_dense*self.scene_extent
        mask = torch.logical_and(mask_grad,mask_scale).squeeze()
        if torch.sum(mask) == 0:
            return
        stds = torch.exp(self.params[2][mask])
        means = torch.zeros(stds.size(0),3)
        samples = torch.normal(mean=means,std=stds)
        rots = build_rotation(self.params[3][mask]).cpu()
        self.params[1][mask] += torch.bmm(rots,samples.unsqueeze(-1)).squeeze(-1)
        self.params[1][mask] += grad_xyz[mask]
        self.params = [torch.vstack([p,p[mask]]) for p in self.params]
        print("CLONE", torch.sum(mask))

    @torch.no_grad()
    def split(self,grad_xyz_abs):
        mask_grad = grad_xyz_abs>self.gradient_threshold
        mask_scale = torch.max(torch.exp(self.params[2]),dim=1).values>self.percent_dense*self.scene_extent
        mask = torch.logical_and(mask_grad,mask_scale).squeeze()
        if torch.sum(mask) == 0:
            return
        stds = torch.exp(self.params[2][mask])
        means = torch.zeros(stds.size(0),3)
        samples = torch.normal(mean=means,std=stds)
        rots = build_rotation(self.params[3][mask]).cpu()
        self.params[1][mask] += torch.bmm(rots,samples.unsqueeze(-1)).squeeze(-1)
        self.params[2][mask] = torch.log(self.params[2][mask]/self.split_scale)
        self.params = [torch.vstack([p,p[mask]]) for p in self.params]
        print("SPLIT", torch.sum(mask))

        
    def train_iteration(self, step: int) -> Dict[str, float]:

        vp_id = self.viewpoint_ids[step%self.viewpoint_ids.shape[0]] 

        print(f'iter {step}, vp_id {vp_id} ---------------------')
        img = torch.from_numpy(self.train_dataset['images'][vp_id][:,:,:3])
        train_cameras = self.train_dataset['cameras']

        cameras_th = train_cameras.apply(lambda x, _: torch.from_numpy(x).contiguous().to(self.device))
        camera_th = cameras_th.__getitem__(vp_id)
        xy = cameras.get_image_pixels(camera_th.image_sizes)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera_th.image_sizes
        
        setup = {
            'tracer':self.tracer,
            'ray_origins':ray_origins.float().squeeze(0).contiguous(),
            'ray_directions':ray_directions.float().squeeze(0).contiguous(),
            'width':res_x,
            'height':res_y,
            'target_img':img.reshape(res_x*res_y,3).to(self.device),
            'sh_deg':self.active_sh_degree
        }
        part_opac_ = torch.sigmoid(self.params[0])
        part_scale_ = torch.exp(self.params[2])
        L,out = trace_function(setup,part_opac_,self.params[1],part_scale_,self.params[3],self.params[4],self.params[5])
        if L < self.best_loss:
            self.best_loss = L.detach()
            self.best_model = [x.detach().clone() for x in self.params]

        def print_stats(param,name):
            print(name,torch.mean(param,dim=0),torch.min(param,dim=0)[0],torch.max(param,dim=0)[0])
        print_stats(self.xyz.detach().cpu(),'xyz')
        print_stats(self.scale.detach().cpu(),'scale')
        print_stats(self.color.detach().cpu(),'color')
        #print_stats(self.sh.detach().cpu(),'sh')
        print_stats(self.opac.detach().cpu(),'opac')

        print(L.detach().item())
        print('------------------')

        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()

        if self.hparams['densify']:    
            self.xyz_grad_abs_acc += torch.linalg.norm(self.params[1].grad,dim=1)
            xyz_grad = self.params[1].grad.clone()

            self.optimizer.zero_grad()

            self.xyz_grad_abs_acc[self.xyz_grad_abs_acc.isnan()] = 0.

            num_orig = self.params[0].shape[0]
            self.clone(xyz_grad, self.xyz_grad_abs_acc)
            num_clone = self.params[0].shape[0]
            grad_padded = torch.zeros(self.params[0].shape[0])
            grad_padded[:self.xyz_grad_abs_acc.shape[0]] = self.xyz_grad_abs_acc
            self.split(grad_padded)
            num_split = self.params[0].shape[0]
            prune_mask = self.prune()
            num_prune = self.params[0].shape[0]

            densified = num_split-num_orig!=0
            pruned = num_split-num_prune!=0
            if densified:
                self.xyz_grad_abs_acc = torch.zeros(self.params[1].shape[0])
            if pruned:
                self.xyz_grad_abs_acc = self.xyz_grad_abs_acc[~prune_mask]
            if densified or pruned:
                for p in self.params:
                    p.requires_grad_()
                self.optimizer = torch.optim.AdamW(self.params,lr=self.lr, weight_decay=1e-4, eps=1e-2)

        return {"mse":L.detach().item(),"vp_id":vp_id, "out_image":out.detach().cpu().reshape(res_y,res_x,3).numpy()}


    @torch.no_grad()
    def render(self, camera : Cameras, *, options=None):

        camera_th = camera.apply(lambda x, _: torch.from_numpy(x).contiguous().to(self.device))
        camera_th = camera_th.__getitem__(0)
        xy = cameras.get_image_pixels(camera_th.image_sizes)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera_th.image_sizes

        time_ms = 0
        nit = 1
        for i in range(nit):
            res = self.tracer.trace_rays(ray_origins.float().squeeze(0).contiguous(),
                                         ray_directions.float().squeeze(0).contiguous(),
                                         res_x, res_y,
                                         False,torch.tensor(0.))
            time_ms += res["time_ms"]
            #print(i,time_ms)
            #print(res["num_its"])
        time_ms /= nit
        
        color = res["radiance"].cpu().reshape(res_y,res_x,3).numpy()
        transmittance = res["transmittance"].cpu().reshape(res_y,res_x)[:,:,None].repeat(1,1,3).numpy()
        debug_map_0 = res["debug_map_0"].cpu().reshape(res_x,res_y,3).numpy()
        debug_map_1 = res["debug_map_1"].cpu().reshape(res_x,res_y,3).numpy()
        time_ms = res["time_ms"]
        num_its = res["num_its"]
        #print(num_its)
        #print(1000/time_ms, num_its/time_ms, res_x,res_y)
        print(time_ms, num_its, res_x,res_y)
        
        return {
            "color": color, #+ transmittance,
            "transmittance": transmittance,
            "debug_map_0": debug_map_0,
            "debug_map_1": debug_map_1,
        }