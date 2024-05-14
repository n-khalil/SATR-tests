import kaolin as kal
import torch

class SPCBuilder:
    def __init__(self, point_cloud, level, device):
        self.point_cloud = point_cloud
        self.level = level
        self.device=device

    def build_SPC(self):
        spc_features = torch.arange(0, self.point_cloud.shape[0]).unsqueeze(1).to(self.device)
        spc = kal.ops.conversions.pointcloud.unbatched_pointcloud_to_spc(
            pointcloud=self.point_cloud, level=self.level, features=spc_features)
        return spc
    
    def __call__(self):
        return self.build_SPC()