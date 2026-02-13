"""ASCII PLY loader"""
import os
import numpy as np


def load_ply(path):
    """Parse ASCII PLY to (N,3) vertices and (M,3) faces."""
    verts, faces = [], []
    in_header = True
    
    with open(path, "r") as f:
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            
            parts = line.split()
            if len(parts) == 5:  # x y z s t -> vertex
                verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif len(parts) == 4 and parts[0] == "3":  # 3 v1 v2 v3 -> face
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                
    return np.array(verts), np.array(faces)


def get_meshes():
    """Load obj1.ply and obj2.ply."""
    base = os.path.dirname(os.path.abspath(__file__))
    v1, f1 = load_ply(os.path.join(base, "obj1.ply"))
    v2, f2 = load_ply(os.path.join(base, "obj2.ply"))
    return v1, f1, v2, f2
