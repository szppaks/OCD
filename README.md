# OCD
OCD very fast point cloud outlier filter

Octree based "radius-outlier-like" point cloud filter
Based on  'Fast Radius Outlier Filter Variant for Large Point Clouds by Péter Szutor and Marianna Zichar'
https://www.mdpi.com/2306-5729/8/10/149


How to use:
Transfer the points to a numpy block.
The numpy array should be passed to the filter, which will return the points that are not outliers. (Caution: it does not keep the original order)

How to install
Use the following modules, these must be piped in first:
numpy
numba

An example of how to use it:
import ocdfilter
import open3d as o3d
import numpy as np
pcd=o3d.io.read_point_cloud('samplepointcloud.pcd',format='pcd')
ppoints=np.asarray(pcd.points)
occ=ocdfilter.obfilter(ppoints,0.5,9,12)
filteredpcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(occ))    
o3d.io.write_point_cloud('filteredpcd.pcd', filteredpcd)
