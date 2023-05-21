import numpy as np
import open3d as o3d


def align_vector_to_another(a, b):
    return normalized(np.cross(a, b))[0] * np.arccos(np.dot(a, b))


def normalized(a, axis=-1):
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    n[n == 0] = 1
    return a / n, n


class LineMesh(object):
    def __init__(self, points, lines=None, colors=(0, 0.8, 0), radius=0.05, smooth_ends=False):
        self.points = np.asarray(points)
        self.lines = np.asarray(lines) if lines else np.array([(i, i + 1) for i in range(len(points) - 1)])
        self.colors = np.asarray(colors)
        self.radius = radius
        self.smooth_ends = smooth_ends
        self.create_line_mesh()

    def create_line_mesh(self):
        start_pts = self.points[self.lines[:, 0]]
        end_pts = self.points[self.lines[:, 1]]
        line_segs = end_pts - start_pts
        unit_line_segs, line_lens = normalized(line_segs)
        self.geom = o3d.geometry.TriangleMesh()
        pt_ids = set()
        for i in range(len(line_segs)):
            if not unit_line_segs[i].any():
                continue
            color = self.colors if self.colors.ndim == 1 else self.colors[i]
            R = align_vector_to_another((0, 0, 1), unit_line_segs[i])
            t = (start_pts[i] + end_pts[i]) / 2
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(self.radius, line_lens[i]).rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(R)).translate(t)
            cylinder.paint_uniform_color(color)
            self.geom += cylinder
            if self.smooth_ends:
                for j in range(2):
                    pt_id = self.lines[i, j]
                    if pt_id not in pt_ids:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(self.radius).translate(self.points[pt_id])
                        sphere.paint_uniform_color(color)
                        self.geom += sphere
                        pt_ids.add(pt_id)


if __name__ == '__main__':
    points = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
    lines = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7))
    colors = ((1, 0, 0),) * len(lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    points = np.array(points) + (0, 0, 2)
    line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
    points = np.array(points) + (0, 2, 0)
    line_mesh2 = LineMesh(points, colors=np.random.random(size=(len(points) - 1, 3)), smooth_ends=True)

    o3d.visualization.draw_geometries((line_set, line_mesh1.geom, line_mesh2.geom))
