import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import menpo3d.io as m3io
import menpo.io as mio
from menpo.shape import PointCloud


def getCentroids(verts):
    tris = Delaunay(verts)
    centroids = []
    for tri in tris.simplices:
        centroids.append([np.sum(verts[tri][:, 0])/3.0, np.sum(verts[tri][:, 1])/3.0])
    centroids = np.asarray(centroids)
    return np.concatenate((verts, centroids), axis=0)


def main():
    vertices = np.load('../data/save-img/blender/vertices_68.npy')
    template = m3io.import_mesh('../data/save-img/blender/base_template.obj')
    # point_cloud = PointCloud(template.points[vertices])
    vertices_all_2d = template.points[:, :2]
    vertices_68_3d = template.points[vertices]

    vertices_68_2d = vertices_68_3d[:, :2]

    vertices_68_2d_new = vertices_68_2d.copy()

    for i in range(0, 2):
        vertices_68_2d_new = getCentroids(vertices_68_2d_new)

    ctree = cKDTree(vertices_all_2d)

    indexs = []
    for vert in vertices_68_2d_new:
        ds, inds = ctree.query(vert, 1)
        # print(ds, inds)
        indexs.append(inds)
    print('***************************************')
    print(len(indexs))
    # ds, inds = ctree.query(vertices_68_2d_new[0], 1)
    np.save('../data/save-img/blender/vertices_dense.npy', np.asarray(indexs))
    # plt.triplot(vertices_2d[:, 0], vertices_2d[:, 1], tri.simplices)
    plt.plot(vertices_68_2d_new[:, 0], vertices_68_2d_new[:, 1], 'ro')

    plt.show()


if __name__ == '__main__':
    main()
