import open3d as o3d
import numpy as np

class PointCloudTriangulator:
    def __init__(self, point_cloud, alpha=0.03):
        self.point_cloud = point_cloud
        self.alpha=alpha
            
    def triangulate_pt_cloud(self):
        o3d_points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd = o3d.geometry.PointCloud(o3d_points)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, self.alpha)
        remeshed_vertices = np.asarray(mesh.vertices)
        remeshed_faces = np.asarray(mesh.triangles)
        return  remeshed_vertices, remeshed_faces
    
    def __call__(self):
        return self.triangulate_pt_cloud()