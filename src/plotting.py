import matplotlib as mpl
import matplotlib.pyplot as plt
import src.device as device

def plot_device_snapshot(dev: device.UnitCellDevice, frame: device.Frame):
    fig,ax = plt.subplots()
    # Draw device
    for x in range(dev.w):
        for y in range(dev.h):
            ax.add_patch(mpl.patches.Rectangle((x+0.05, y+0.05), 0.9, 0.9, color='w', edgecolor='k'))
    
    for q,coords in frame.qubit_positions.items():
        