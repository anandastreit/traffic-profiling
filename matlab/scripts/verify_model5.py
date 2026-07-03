import h5py, numpy as np, pandas as pd, os

BASE = r'D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus'
CSV_DIR = os.path.join(BASE, r'tables\models\train\2019-08-19_2019-09-22')

mats = {
    'noWeights':        os.path.join(BASE, 'models_1-08_trafficDownUp_2019-08-19_2019-09-22_noWeights.mat'),
    'filter_NaNs':      os.path.join(BASE, 'models_1-08_trafficDownUp_2019-08-19_2019-09-22_noWeights_filter_NaNs.mat'),
    'filter_NaNs_v2':   os.path.join(BASE, 'models_1-08_trafficDownUp_2019-08-19_2019-09-22_noWeights_filter_NaNs_v2.mat'),
}

csvs = {
    'noWeights': {
        'A': os.path.join(CSV_DIR, 'model5_mode1_trafficDownUp_train_noWeights.csv'),
        'B': os.path.join(CSV_DIR, 'model5_mode2_trafficDownUp_train_noWeights.csv'),
        'C': os.path.join(CSV_DIR, 'model5_mode3_trafficDownUp_train_noWeights.csv'),
    },
    'filter_NaNs': {
        'A': os.path.join(CSV_DIR, 'model5_mode1_trafficDownUp_train_noWeights_filter_NaNs.csv'),
        'B': os.path.join(CSV_DIR, 'model5_mode2_trafficDownUp_train_noWeights_filter_NaNs.csv'),
        'C': os.path.join(CSV_DIR, 'model5_mode3_trafficDownUp_train_noWeights_filter_NaNs.csv'),
    },
}

def get_model5_from_mat(filepath):
    with h5py.File(filepath, 'r') as f:
        top = [k for k in f.keys() if not k.startswith('#')]
        print(f'  Top-level keys: {top}')
        if 'Model5' not in f:
            print('  Model5 NAO encontrado')
            return None, None, None
        refs = f['Model5']  # (3,1) object array de refs
        A = np.array(f[refs[0, 0]], dtype=float).T
        B = np.array(f[refs[1, 0]], dtype=float).T
        C = np.array(f[refs[2, 0]], dtype=float).T
        print(f'  Model5: A={A.shape} B={B.shape} C={C.shape}')
        return A, B, C

def compare(mat_arr, csv_path, label):
    if not os.path.exists(csv_path):
        print(f'    {label}: CSV nao existe ({os.path.basename(csv_path)})')
        return
    csv_arr = pd.read_csv(csv_path, header=None).values
    if mat_arr.shape != csv_arr.shape:
        print(f'    {label}: shapes diferentes! mat={mat_arr.shape} csv={csv_arr.shape}')
        return
    max_diff = np.max(np.abs(mat_arr - csv_arr))
    match = np.allclose(mat_arr, csv_arr, atol=1e-6)
    print(f'    {label}: shape={mat_arr.shape}  max_diff={max_diff:.2e}  igual={match}')

for mat_name, mat_path in mats.items():
    print(f'\n=== {mat_name} ===')
    A, B, C = get_model5_from_mat(mat_path)
    if A is None:
        continue
    for csv_name, csv_dict in csvs.items():
        print(f'  vs CSV ({csv_name}):')
        compare(A, csv_dict['A'], 'A')
        compare(B, csv_dict['B'], 'B')
        compare(C, csv_dict['C'], 'C')
