"""
nohup python -m notebooks.10_comparison_fixedT2 > notebooks/10_out.txt 2>&1 &
"""
import pickle
import sinter
from src.decoders import BPOSD

if __name__ == '__main__':
    with open('notebooks/data/10_tasks.pkl', 'rb') as f:
        data = pickle.load(f)
        tasks = data['tasks']

    custom_decoders = {'bposd': BPOSD(
        max_iter=1000,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        osd_method='OSD_CS',
        osd_order=10,
    )}

    stats = sinter.collect(
        num_workers=24,
        tasks=tasks,
        max_shots=10**4,
        max_errors=100,
        custom_decoders=custom_decoders,
        print_progress=True,
        save_resume_filepath='notebooks/data/10_results_TMP',
    )

    with open('notebooks/data/10_results.pkl', 'wb') as f:
        pickle.dump(stats, f)