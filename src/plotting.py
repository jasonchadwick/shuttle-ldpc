import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import src.device as device
from src.QECCode import QECCode

def plot_device_snapshot(dev: device.UnitCellDevice, code: QECCode, frame: device.Frame, ax = None):
    if ax is None:
        fig,ax = plt.subplots(figsize=(6,6))
    ax.clear()
    ax.add_patch(mpl.patches.Rectangle((-1.05, -1.05), dev.w+0.1, dev.h+0.1, facecolor='lightgray'))
    # Draw device
    for x in range(dev.w):
        for y in range(dev.h):
            ax.add_patch(mpl.patches.Rectangle((x-1+0.05, y-1+0.05), 0.9, 0.9, facecolor='w', edgecolor='none'))
            if x < dev.w and y < dev.h:
                ax.add_patch(mpl.patches.Rectangle((x-0.5, y-0.25), 0.2, 0.2, facecolor='orange', edgecolor='k'))
                ax.add_patch(mpl.patches.Rectangle((x-0.25, y-0.45), 0.2, 0.4, facecolor='green', edgecolor='k'))
    
    for q,coords in frame.qubit_positions.items():
        if q in code.data_indices:
            color = 'k'
        elif q in code.Z_ancilla_indices:
            color = 'blue'
        elif q in code.X_ancilla_indices:
            color = 'red'
        else:
            print('Unknown qubit', q)
            color = 'purple'
        ax.add_patch(mpl.patches.Circle(coords, radius=0.1, color=color))

    seen_coords = set()
    for qbs in frame.twoq_gates:
        coords = frame.qubit_positions[qbs[0]]
        x = int(round(coords[0]))
        y = int(round(coords[1]))
        coords = (x,y)
        if coords not in seen_coords:
            ax.add_patch(mpl.patches.Rectangle((x-0.25, y-0.45), 0.2, 0.4, edgecolor='limegreen', facecolor='none', linewidth=2))
        seen_coords.add(coords)

    ax.set_xlim(-2, dev.w)
    ax.set_ylim(-2, dev.h)
    ax.set_title(f't={frame.t}')
    ax.set_axis_off()
    ax.set_aspect('equal')
    return ax

def animate_device(
        dev: device.UnitCellDevice,
        code: QECCode,
        frames: list[device.Frame],
        filename: str | None = None,
        fps: int = 10,
    ):
    def update(i, ax):
        ax = plot_device_snapshot(dev, code, frames[i], ax)

    fig,ax = plt.subplots()
    ani = animation.FuncAnimation(fig=fig, func=lambda i: update(i, ax), frames=len(frames), interval=30)

    if filename:
        ani.save(filename, writer='imagemagick', fps=fps)
    return ani