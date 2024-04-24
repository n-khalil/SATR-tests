import open3d as o3d
import numpy as np

class PointCloudTriangulator:
    def __init__(self):
        pass
    
    def triangulate_pt_cloud(self, point_cloud, alpha=0.03):
        o3d_points = o3d.utility.Vector3dVector(point_cloud)
        pcd = o3d.geometry.PointCloud(o3d_points)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        remeshed_vertices = np.asarray(mesh.vertices)
        remeshed_faces = np.asarray(mesh.triangles)
        return  remeshed_vertices, remeshed_faces