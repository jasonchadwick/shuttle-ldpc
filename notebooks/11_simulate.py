"""
nohup python -m notebooks.11_simulate > notebooks/11_out.txt 2>&1 &
"""
import pickle
import sinter
from src.decoders import BPOSD

if __name__ == '__main__':
    with open('notebooks/data/11_tasks.pkl', 'rb') as f:
        data = pickle.load(f)
        tasks = data['tasks']

    custom_decoders = {'bposd': BPOSD(
        max_iter=1000,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        osd_method='OSD_CS',
        osd_order=10,
    )}

    tasks_surface = [t for t in tasks if t.json_metadata['type'] == 'Surface']
    tasks_other = [t for t in tasks if t.json_metadata['type'] != 'Surface']

    stats_surface = sinter.collect(
        num_workers=40,
        tasks=tasks_surface,
        max_shots=10**7,
        max_errors=500,
        print_progress=True,
        save_resume_filepath='notebooks/data/11_results_TMP',
    )

    stats = sinter.collect(
        num_workers=40,
        tasks=tasks_other,
        max_shots=2*10**4,
        max_errors=100,
        custom_decoders=custom_decoders,
        print_progress=True,
        save_resume_filepath='notebooks/data/11_results_TMP',
    )

    with open('notebooks/data/11_results.pkl', 'wb') as f:
        pickle.dump(stats, f)