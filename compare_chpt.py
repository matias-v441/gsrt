#%%
import torch
import lovely_tensors as lt
lt.monkey_patch()
chpt_iter = 700
chpt_grut = f'/home/matbi/proj/3dgrut/runs/lego-2204_020424/ours_{chpt_iter}/ckpt_{chpt_iter}.pt'
chpt_my = f'/home/matbi/proj/gsrt/gsrt_checkpoint/checkpoint_{chpt_iter}.pt'

class Data:
    def __init__(self,chpt,type=''):
        self.checkpoint = chpt
        if type == "grut":
            self.load_3dgrt_checkpoint()
        else:
            self.load_checkpoint()

    def load_checkpoint(self):
        ch = torch.load(self.checkpoint)
        self._xyz = ch['xyz']
        self._scaling = ch['scaling']
        self._rotation = ch['rotation']
        self._opacity = ch['opacity']
        self._features_dc = ch['f_dc']
        self._features_rest = ch['f_rest']
        self.active_sh_degree = ch['sh_deg']
        self.max_sh_degree = 3
        self._color = ch['color']
        self._resp = torch.zeros_like(self._opacity)

    def load_3dgrt_checkpoint(self):
        print(self.checkpoint)
        ch = torch.load(self.checkpoint)
        self._xyz = ch['positions'].detach()
        self._scaling = ch['scale'].detach()
        self._rotation = ch['rotation'].detach()
        self._opacity = ch['density'].detach()
        self._features_dc = ch['features_albedo'].detach().reshape(self._xyz.shape[0],1,3) #torch.zeros((self._xyz.shape[0],1,3)).float().cuda()
        self._features_rest = ch['features_specular'].detach().reshape(self._xyz.shape[0],15,3) #torch.zeros((self._xyz.shape[0],15,3)).float().cuda()
        self.active_sh_degree = ch['n_active_features']
        self.max_sh_degree = ch['max_n_features']
        self._color = torch.zeros_like(self._xyz) #ch['color']
        self._resp = torch.zeros_like(self._opacity)
        self.ref_grads = torch.load(self.checkpoint.replace(".pt","_grad.pt"))

    def compare(self, o):
        def comp(v1,v2):
            if (v1,v2) != (None,None) and v1.shape == v2.shape:
                print(torch.all(torch.isclose(v1,v2)).item(), 
                torch.norm(v1-v2,dim=-1))
            print(v1)
            print(v2)
        for attr in self.__dict__.keys():
            v1,v2 = self.__getattribute__(attr),o.__getattribute__(attr)
            if type(v1) is torch.Tensor:
                print(attr)
                comp(v1,v2) 

data_grut = Data(chpt_grut,type='grut')
data_my = Data(chpt_my)
data_my.compare(data_grut)