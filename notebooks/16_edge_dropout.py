"""
nohup python -m notebooks.16_edge_dropout > notebooks/16_out.txt 2>&1 &
"""

import numpy as np
from sympy.abc import x, y

# from qldpc import codes
import networkx as nx
import matplotlib.pyplot as plt
import stim
import pickle
import sinter
from copy import deepcopy
import scipy
import signal

import src.device as device
import src.plotting as plotter
import src.stim_dag as stim_dag

from src.RotatedSurfaceCode import RotatedSurfaceCode
from src.TileCode import TileCode
from src.HGPCode import HGPCode
from src.SHYPSCode import SHYPSCode
from src.RadialCode import RadialCode
from src.GBCode import GBCode
from src.HypercubeCode import HypercubeCode
from src.decoders import BPOSD
from src.constructor import construct_code

if __name__ == '__main__':
    dropout_rates = list(np.geomspace(1e-4, 1e-1, 100))
    trials = 20
    ds_surface = [3, 5, 7, 9, 11]
    hwp = device.default_hwp

    schedule_lengths_baseline = np.zeros(len(ds_surface), int)
    for i_d,d in enumerate(ds_surface):
        code = RotatedSurfaceCode(d)
        buffer = 3
        xmax,ymax = 0,0
        data_coords = dict()
        for i in code.data_indices:
            x,y = code.qubit_coords[i]
            data_coords[i] = (x+buffer, y+buffer)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
        dev = device.UnitCellDevice(xmax+2*buffer, ymax+2*buffer, hwp)
        sched = dev.compile_QEC_schedule(
            code,
            data_coords,
            code.check_cx_layers,
            rounds=1,
            use_highways=False,
            refocus_shuttle_noise=True,
            separate_X_Z=False,
            use_cache=False,
            suppress_printing=True,
        )
        schedule_lengths_baseline[i_d] = sched.total_duration()
    def handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, handler)

    timeout = 60
    rng = np.random.default_rng(0)

    schedule_lengths = np.zeros((len(ds_surface), len(dropout_rates), trials), int)
    for i_d,d in enumerate(ds_surface):
        print(d, end='')
        code = RotatedSurfaceCode(d)
        buffer = 3
        xmax,ymax = 0,0
        data_coords = dict()
        for i in code.data_indices:
            x,y = code.qubit_coords[i]
            data_coords[i] = (x+buffer, y+buffer)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
        dev = device.UnitCellDevice(xmax+2*buffer, ymax+2*buffer, hwp)
        for i_r,dropout_rate in enumerate(dropout_rates):
            for trial in range(trials):
                dropped_edges = [(((int(c00), int(c01))), (int(c10), int(c11))) for ((c00,c01), (c10,c11)) in rng.choice(dev.all_edges, int(len(dev.all_edges)*dropout_rate))]
                dev.broken_edges = dropped_edges

                try:
                    signal.alarm(timeout)
                    sched = dev.compile_QEC_schedule(
                        code,
                        data_coords,
                        code.check_cx_layers,
                        rounds=1,
                        use_highways=False,
                        refocus_shuttle_noise=True,
                        separate_X_Z=False,
                        use_cache=False,
                        suppress_printing=True,
                    )
                    print('.', end='', flush=True)
                    schedule_lengths[i_d,i_r,trial] = sched.total_duration()
                except RuntimeError as e:
                    print('X', end='', flush=True)
                    schedule_lengths[i_d,i_r,trial] = 10**10
                except TimeoutError as e:
                    print('T', end='', flush=True)
                    schedule_lengths[i_d,i_r,trial] = 10**10
                signal.alarm(0)
            if trials > 10:
                print('|', end='', flush=True)
        print()

    print('Saving to file...')

    with open('notebooks/data/16_dropouts.pkl', 'wb') as f:
        pickle.dump({
            'ds': ds_surface,
            'dropout_rates':dropout_rates,
            'trials':trials,
            'schedule_lengths_baseline': schedule_lengths_baseline,
            'schedule_lengths': schedule_lengths,
        }, f)