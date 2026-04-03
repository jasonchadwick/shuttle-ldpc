import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import src.device as device
from src.QECCode import QECCode

plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#0072B2', '#CC79A7', '#009E73', '#E69F00', '#56B4E9', '#D55E00', '#F0E442']) 
plt.rcParams['font.family'] = 'serif'

def interpolate(color1, color2, alpha):
    return LinearSegmentedColormap.from_list('_', [color1, color2])(alpha)

def get_q_color(q: int, code: QECCode):
    if q in code.data_indices:
        color = 'k'
    elif q in code.Z_ancilla_indices:
        color = 'C0'
    elif q in code.X_ancilla_indices:
        color = 'C1'
    else:
        print('Unknown qubit', q)
        color = 'C5'
    return color

def plot_device_snapshot(
        dev: device.UnitCellDevice,
        code: QECCode,
        frame: device.Frame,
        ax = None,
    ):
    if ax is None:
        fig,ax = plt.subplots(figsize=(6,6))
    ax.clear()
    ax.add_patch(mpl.patches.Rectangle((-1.05, -1.05), dev.w+0.1, dev.h+0.1, facecolor=interpolate('C0', 'w', 0.6), zorder=0.3))
    # Draw device
    for x in range(dev.w):
        for y in range(dev.h):
            ax.add_patch(mpl.patches.Rectangle((x-1+0.05, y-1+0.05), 0.9, 0.9, facecolor='w', edgecolor='none', zorder=0.4))
            if x < dev.w and y < dev.h:
                ax.add_patch(mpl.patches.Rectangle((x-0.5, y-0.25), 0.2, 0.2, facecolor='C5', edgecolor='none', zorder=0.5))
                ax.add_patch(mpl.patches.Rectangle((x-0.25, y-0.45), 0.2, 0.4, facecolor='C3', edgecolor='none', zorder=0.5))
    
    for q,coords in frame.qubit_positions.items():
        color = get_q_color(q, code)
        fc = interpolate(color, 'w', 0)
        color = interpolate(color, 'k', 0.2)
        ax.add_patch(mpl.patches.Circle(coords, radius=0.15, facecolor=fc, edgecolor=color, linewidth=1, zorder=2))

    seen_coords = set()
    for qbs in frame.twoq_gates:
        coords = frame.qubit_positions[qbs[0]]
        x = int(round(coords[0]))
        y = int(round(coords[1]))
        coords = (x,y)
        if coords not in seen_coords:
            ax.add_patch(mpl.patches.Rectangle((x-0.3, y-0.5), 0.3, 0.5, edgecolor='none', facecolor='C2', alpha=0.5, zorder=3))
            ax.add_patch(mpl.patches.Rectangle((x-0.3, y-0.5), 0.3, 0.5, edgecolor='C2', facecolor='none', linewidth=2, zorder=3))
        seen_coords.add(coords)

    ax.set_xlim(-2, dev.w)
    ax.set_ylim(-2, dev.h)
    ax.set_title(f't={frame.t}ns')
    ax.set_axis_off()
    ax.set_aspect('equal')
    return ax

def plot_transition(
        dev: device.UnitCellDevice,
        code: QECCode,
        frames: list[device.Frame],
        ax = None,
        highlight_qubits: list[int] = [],
    ):
    qubit_loc_history: dict[int, list[tuple[float, float]]] = dict()
    for f in frames:
        for q,pos in f.qubit_positions.items():
            if q not in qubit_loc_history:
                qubit_loc_history[q] = [pos]
            else:
                if qubit_loc_history[q][-1] != pos:
                    qubit_loc_history[q].append(pos)
    
    ax = plot_device_snapshot(dev, code, frames[-1], ax)

    seen_coords = set()
    for frame in frames:
        for qbs in frame.twoq_gates:
            coords = frame.qubit_positions[qbs[0]]
            x = int(round(coords[0]))
            y = int(round(coords[1]))
            coords = (x,y)
            if coords not in seen_coords:
                ax.add_patch(mpl.patches.Rectangle((x-0.3, y-0.5), 0.3, 0.5, edgecolor='none', facecolor='C2', alpha=0.5, zorder=3))
                ax.add_patch(mpl.patches.Rectangle((x-0.3, y-0.5), 0.3, 0.5, edgecolor='C2', facecolor='none', linewidth=2, zorder=3))
            seen_coords.add(coords)

    if not highlight_qubits:
        highlight_qubits = list(qubit_loc_history.keys())

    for q,positions in qubit_loc_history.items():
        if len(positions) == 1 or q in code.data_indices or q not in highlight_qubits:
            continue
        color = interpolate(get_q_color(q, code), 'k', 0.5)
        # color = 'k'
        for i,pos in enumerate(positions[:-2]):
            pos2 = positions[i+1]
        ax.plot([x for x,y in positions[:-1]], [y for x,y in positions[:-1]], linestyle='-', color='w', marker='none', linewidth=3, zorder=2.4)
        ax.plot([x for x,y in positions[:-1]], [y for x,y in positions[:-1]], linestyle='-', color=color, marker='none', linewidth=2, zorder=2.5)
        pos = positions[-2]
        pos2 = positions[-1]
        ax.arrow(pos[0], pos[1], pos2[0]-pos[0], pos2[1]-pos[1], color='w', head_width=0.15, head_length=0.095, length_includes_head=True, linewidth=3, zorder=2.4)
        ax.arrow(pos[0], pos[1], pos2[0]-pos[0], pos2[1]-pos[1], color=color, head_width=0.13, head_length=0.09, length_includes_head=True, linewidth=2, zorder=2.5)
        # ax.annotate('', xytext=pos, xy=pos2, arrowprops=dict(arrowstyle='->,
        # head_width=0.5, head_length=0.35', linewidth=2, color=color),
        # zorder=2.5)
    
    ax.set_title(f't={frames[0].t} to {frames[-1].t} (dt={frames[-1].t-frames[0].t})')
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