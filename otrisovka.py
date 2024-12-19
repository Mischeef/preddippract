import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def read_off(file_path):
    verts = []
    faces = []
    with open(file_path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, _ = map(int, file.readline().strip().split())
        for _ in range(n_verts):
            verts.append(list(map(float, file.readline().strip().split())))
        for _ in range(n_faces):
            parts = list(map(int, file.readline().strip().split()))
            if parts[0] != 3:
                raise ValueError('Only triangular faces are supported')
            faces.append(parts[1:])
    return np.array(verts), np.array(faces)

def plot_off(verts, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    poly3d = Poly3DCollection([verts[face] for face in faces], alpha=0.5, edgecolor='k')
    ax.add_collection3d(poly3d)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Пример использования
file_path = r'C:\Users\gerpv\Desktop\predpp\preddippract\ModelNet10\bed\test\bed_0517.off'
verts, faces = read_off(file_path)
plot_off(verts, faces)