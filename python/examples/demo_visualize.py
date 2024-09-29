import os
import open3d as o3d
import numpy as np
import pypatchworkpp

cur_dir = os.path.dirname(os.path.abspath(__file__))
input_cloud_filepath = os.path.join(cur_dir, '../../data/000003.bin')


def read_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    return scan

if __name__ == "__main__":

    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
    params.verbose = True

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    # Load point cloud
    pointcloud = read_bin(input_cloud_filepath)

    # Estimate Ground
    PatchworkPLUSPLUS.estimateGround(pointcloud)

    # Get Ground and Nonground
    ground      = PatchworkPLUSPLUS.getGround()
    nonground   = PatchworkPLUSPLUS.getNonground()
    time_taken  = PatchworkPLUSPLUS.getTimeTaken()

    ground_idx      = PatchworkPLUSPLUS.getGroundIndices()
    nonground_idx   = PatchworkPLUSPLUS.getNongroundIndices()

    # Get centers and normals for patches
    centers     = PatchworkPLUSPLUS.getCenters()
    normals     = PatchworkPLUSPLUS.getNormals()

    print("Original Points  #: ", pointcloud.shape[0])
    print("Ground Points    #: ", ground.shape[0])
    print("Nonground Points #: ", nonground.shape[0])
    print("Time Taken : ", time_taken / 1000000, "(sec)")
    print("Press ... \n")
    print("\t H  : help")
    print("\t N  : visualize the surface normals")
    print("\tESC : close the Open3D window")

    # Visualize
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width = 600, height = 400)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    ground_o3d = o3d.geometry.PointCloud()
    ground_o3d.points = o3d.utility.Vector3dVector(ground)
    ground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float) # RGB
    )

    nonground_o3d = o3d.geometry.PointCloud()
    nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
    nonground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float) #RGB
    )


    PLOT_HULL = True

    if True:
        if PLOT_HULL:
            import numpy as np
            from sklearn.cluster import DBSCAN
            from scipy.spatial import ConvexHull
            from shapely.geometry import Polygon, MultiPolygon
            from concave_hull import concave_hull, concave_hull_indexes
            import alphashape
            from simplification.cutil import (
                simplify_coords,
                simplify_coords_idx,
                simplify_coords_vw,
                simplify_coords_vw_idx,
                simplify_coords_vwp,
            )

            nonground_xy = nonground[:, :2]
            dbscan = DBSCAN(eps=0.3, min_samples=3)
            cluster_labels = dbscan.fit_predict(nonground_xy)

            polygons = []

            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_points = nonground_xy[cluster_labels == cluster_id]
                if len(cluster_points) >= 3:  # Convex hull requires at least 3 points
                    idxes = concave_hull_indexes(
                        cluster_points,
                        concavity=2  # TODO: check
                    )
                    # breakpoint()
                    coords = cluster_points[idxes]
                    simplified_coords = simplify_coords(coords, 0.1)
                    
                    if len(simplified_coords) < 4:
                        simplified_coords = coords

                    polygon = Polygon(simplified_coords)
                    # polygon = Polygon(coords)
                    polygons.append(polygon)

                    # concave_hull = alphashape.alphashape(cluster_points, alpha=0.2) 

                    # if isinstance(concave_hull, Polygon):
                    #     polygons.append(concave_hull)
                    # elif isinstance(concave_hull, MultiPolygon):
                    #     polygons.extend([poly for poly in concave_hull.geoms])



        import numpy as np
        import tkinter
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 14))
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)

        if PLOT_HULL:
            # plt.scatter(nonground_xy[:, 0], nonground_xy[:, 1], c=clusters, cmap='tab10', marker='o', label="Points")

            for polygon in polygons:
                x, y = polygon.exterior.xy
                plt.fill(x, y, alpha=0.3, edgecolor='black', facecolor='cyan', label="Polygon")
        else:
            plt.scatter(nonground[:, 0], nonground[:, 1], color='red', label='nonground', s=1)
            plt.scatter(ground[:, 0], ground[:, 1], color='blue', label='ground', s=1)


        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Ground/Noground')

        plt.show()

        exit(1)
        # breakpoint()

    centers_o3d = o3d.geometry.PointCloud()
    centers_o3d.points = o3d.utility.Vector3dVector(centers)
    centers_o3d.normals = o3d.utility.Vector3dVector(normals)
    centers_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float) #RGB
    )

    vis.add_geometry(mesh)
    vis.add_geometry(ground_o3d)
    vis.add_geometry(nonground_o3d)
    vis.add_geometry(centers_o3d)

    vis.run()
    vis.destroy_window()
