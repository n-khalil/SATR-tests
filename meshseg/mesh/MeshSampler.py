import trimesh
import numpy as np
import torch
from collections import defaultdict
import kaolin as kal

class MeshSampler:
    def __init__(self, mesh, device, n_samples_factor=2):
        self.mesh = mesh
        self.n_samples_factor = n_samples_factor
        self.device = device

    def sample_mesh(self, n_samples):
        trimeshMesh = trimesh.Trimesh(self.mesh.vertices.cpu().numpy(), self.mesh.faces.cpu().numpy())
        point_cloud, pt_to_face = trimesh.sample.sample_surface_even(trimeshMesh, n_samples)
        print(f'Sampled {point_cloud.shape[0]} points')
        torchPC = torch.tensor(point_cloud, device=self.device, dtype=torch.float32)
        face_to_all_pts = defaultdict(list)
        for pt in range(len(point_cloud)):
            face_to_all_pts[pt_to_face[pt]].append(pt)
        return torchPC, pt_to_face, face_to_all_pts


    def computeNormals(self, pt_to_face):
        face_normals = kal.ops.mesh.face_normals(
            kal.ops.mesh.index_vertices_by_faces(self.mesh.vertices.unsqueeze(0), self.mesh.faces),
              unit=True).squeeze()
        n_pts = len(pt_to_face)
        pts_normals = torch.zeros((n_pts, 3), device=self.device)
        for pt in range(n_pts):
            pts_normals[pt] = face_normals[pt_to_face[pt]]
        return pts_normals

    def __call__(self):
        n_samples = int(self.mesh.vertices.shape[0] * self.n_samples_factor)
        torchPC, pt_to_face, face_to_all_pts = self.sample_mesh(n_samples)
        pts_normals = self.computeNormals(pt_to_face)
        return torchPC, pt_to_face, face_to_all_pts, pts_normals
