import trimesh
import numpy as np
import torch

class MeshSampler:
    def __init__(self, mesh, device, n_samples_factor=2):
        self.mesh = mesh
        self.n_samples_factor = n_samples_factor
        self.device = device

    def sample_mesh(self, n_samples):
        trimeshMesh = trimesh.Trimesh(self.mesh.vertices.cpu().numpy(), self.mesh.faces.cpu().numpy())
        point_cloud, pt_to_face = trimesh.sample.sample_surface_even(trimeshMesh, n_samples)
        torchPC = torch.tensor(point_cloud, device=self.device, dtype=torch.float32)
        return torchPC, pt_to_face

    def __call__(self):
        n_samples = int(self.mesh.vertices.shape[0] * self.n_samples_factor)
        return self.sample_mesh(n_samples)
