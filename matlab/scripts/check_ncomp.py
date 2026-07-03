import h5py, os

files = [
    r'D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus\models_1-08_trafficDownUp_2019-08-19_2019-09-22_noWeights_filter_NaNs.mat',
    r'D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus\models_1-08_trafficDownUp_2019-08-19_2019-09-22_noWeights.mat',
    r'D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus\weekModels_1e-08_trafficDownUp_2019-08-19_2019-09-22_noWeights_filter_NaNs.mat',
    r'D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus\weekModels_1e-08_trafficDownUp_2019-08-19_2019-09-22_noWeights.mat',
]

for fp in files:
    name = os.path.basename(fp)
    try:
        with h5py.File(fp, 'r') as f:
            top = [k for k in f.keys() if not k.startswith('#')]
            print(name)
            if 'Dataset' in f:
                ds_keys = sorted([k for k in f['Dataset'].keys() if k.startswith('nComp')])
                print(f'  Dataset nComp keys: {ds_keys}')
            week_keys = [k for k in top if k.startswith('Week')]
            if week_keys:
                w = f[week_keys[0]]
                if 'filter_NaNs' in w:
                    fn_keys = sorted([k for k in w['filter_NaNs'].keys() if k.startswith('Model')])
                    print(f'  {week_keys[0]}.filter_NaNs Model keys: {fn_keys}')
                else:
                    sub = sorted([k for k in w.keys() if k.startswith('Model')])
                    print(f'  {week_keys[0]} Model keys: {sub}')
            print(f'  Top-level: {top}')
    except Exception as e:
        print(f'{name}: ERRO - {e}')
    print()
