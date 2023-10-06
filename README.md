# OCD
OCD very fast point cloud outlier filter
<br />
Octree based "radius-outlier-like" point cloud filter<br />
Based on  'Fast Radius Outlier Filter Variant for Large Point Clouds by Péter Szutor and Marianna Zichar'<br />
https://www.mdpi.com/2306-5729/8/10/149<br />


How to use:<br />
Transfer the points to a numpy block.<br />
The numpy array should be passed to the filter, which will return the points that are not outliers. (Caution: it does not keep the original order)

How to install<br />
Use the following modules, these must be piped in first:<br />
numpy<br />
numba<br />

An example of how to use it:<br /><i>
import ocdfilter<br />
import open3d as o3d<br />
import numpy as np<br />
pcd=o3d.io.read_point_cloud('samplepointcloud.pcd',format='pcd')<br />
ppoints=np.asarray(pcd.points)<br />
occ=ocdfilter.obfilter(ppoints,0.5,9,12)<br />
filteredpcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(occ))   <br />
o3d.io.write_point_cloud('filteredpcd.pcd', filteredpcd)<br />
