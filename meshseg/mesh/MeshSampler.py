import trimesh
import open3d as o3d

class MeshSampler:
    def __init__(self, n_samples_factor=2):
        self.mesh = mesh
        self.n_samples_factor = n_samples_factor

    def sample_mesh(self, n_samples):
        trimeshMesh = trimesh.Trimesh(self.mesh.vertices.cpu().numpy(), self.mesh.faces.cpu().numpy())
        self.point_cloud = trimesh.sample.sample_surface_even(trimeshMesh, n_samples)[0]
        return self.point_cloud

    def __call__(self):
        n_samples = int(self.mesh.vertices.shape[0] * self.n_samples_factor)
        point_cloud = self.sample_mesh(n_samples)
        return self.point_cloud
