"""ICP Visualizer - loads PLY meshes, runs ICP, saves a GIF."""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from meshes import get_meshes
from icp import icp


def random_transform(pts, max_angle=np.pi, max_t=3.0, seed=None):
    """Apply a random rigid transform to a point cloud."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-max_angle, max_angle, 3)
    cx, cy, cz = np.cos(a)
    sx, sy, sz = np.sin(a)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    R = Rz @ Ry @ Rx
    t = rng.uniform(-max_t, max_t, (1, 3))
    return (R @ pts.T).T + t


def capture_frame(fig):
    """Render figure to an in-memory PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def main():
    va, fa, vb, fb = get_meshes()
    source = random_transform(va, max_angle=np.pi / 2, max_t=4.0)

    # --- Figure setup ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("ICP — Iterative Closest Point", fontsize=14, fontweight="bold")

    all_pts = np.vstack([source, vb])
    m = 2.0
    ax.set_xlim(all_pts[:,0].min() - m, all_pts[:,0].max() + m)
    ax.set_ylim(all_pts[:,1].min() - m, all_pts[:,1].max() + m)
    ax.set_zlim(all_pts[:,2].min() - m, all_pts[:,2].max() + m)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # Points
    sc_a = ax.scatter(*source.T, c="dodgerblue", s=4, alpha=0.8, label="Source")
    ax.scatter(*vb.T, c="tomato", s=4, alpha=0.5, label="Target")

    # Faces (wireframe)
    faces_on = [True]
    fc_a = Poly3DCollection(source[fa], alpha=0.08, facecolor="dodgerblue", edgecolor="dodgerblue", linewidths=0.3)
    fc_b = Poly3DCollection(vb[fb], alpha=0.06, facecolor="tomato", edgecolor="tomato", linewidths=0.3)
    ax.add_collection3d(fc_a)
    ax.add_collection3d(fc_b)

    # Legend bottom-right, status text top-left — no overlap
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    status = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=10, va="top")

    # W key toggles wireframe (F is reserved for fullscreen by matplotlib)
    def on_key(event):
        if event.key == "w":
            faces_on[0] = not faces_on[0]
            fc_a.set_visible(faces_on[0])
            fc_b.set_visible(faces_on[0])
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Run ICP + capture frames ---
    frames = []
    plt.draw(); plt.pause(0.01)
    frames.append(capture_frame(fig))

    print("Starting ICP …")
    print(f"{'Iter':>5}  {'Mean Error':>12}")
    print("-" * 20)

    for s in icp(source, vb, max_iter=100, tol=1e-7):
        pts = s["transformed"]
        print(f"{s['iteration']:5d}  {s['error']:12.6f}")

        sc_a._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        fc_a.set_verts(pts[fa])

        tag = "  ✓ Converged!" if s["converged"] else ""
        status.set_text(f"Iteration {s['iteration']}  |  Error: {s['error']:.6f}{tag}")

        plt.draw(); plt.pause(0.05)
        frames.append(capture_frame(fig))

    print("-" * 20)

    # --- Save GIF ---
    gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icp_result.gif")
    frames += [frames[-1]] * 15  # linger on final frame
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(f"\nGIF saved to: {gif_path}")
    print("Close the plot window to exit (W = toggle wireframe).")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
