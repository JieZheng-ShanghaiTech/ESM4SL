from subprocess import call
import os
import pandas as pd
import random
import itertools
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

output_root = '/sharedata/home/daihzh/protein/ESM4SL/output/mlp'
cell_lines = ['MELJUSO']
# cell_lines = ['A549', 'IPC298', 'OVCAR8', '22RV1', '293T', 'GI1', 'HELA', 'HS936T', 'HS944T', 'HSC5', 'JURKAT', 'K562', 'MEL202', 'MEWO', 'PC9', 'PK1', 'RPE1', 'MELJUSO', 'PATU8988S']  # 
scenes = ['C2']  #'C1', , 'Tail'

for cl, scene in list(itertools.product(cell_lines, scenes)):
    data_root = f'/sharedata/home/daihzh/protein/ESM4SL/data/SLKB/specific/{cl}/{scene}'
    if not os.path.exists(data_root):
        continue

    print(f'======================== {cl}/{scene} ========================')
    table_path = f'{output_root}/{cl}/{scene}'
    os.makedirs(table_path, exist_ok=True)

    column_names = ['Accuracy', 'Precision', 'Recall', 'AUC', 'AUPR', 'F1', 'BACC']
    metric_df = pd.DataFrame(0, index=range(6), columns=column_names)

    for test_fold in range(5):
        print(f'======================== Fold {test_fold} ========================')

        seed = 42 #random.randint(1, 1000)
        call(['python', 'train.py', 
            '--config-file', 'esm4sl/configuration/mlp/new.yaml',
            '--num-gpus', '1',
            'SEED', str(seed),
            'OUTPUT_DIR', f'{table_path}/fold_{test_fold}',
            'DATASET.TEST_FOLD', str(test_fold),
            'DATASET.TRAIN_FILE', f'{data_root}/sl_train_{test_fold}.csv',
            'DATASET.VAL_FILE', f'{data_root}/sl_val_{test_fold}.csv',
            'DATASET.TEST_FILE', f'{data_root}/sl_test_{test_fold}.csv',
            ])

        newest_version = len(os.listdir(f'{table_path}/fold_{test_fold}/csv_log')) - 1
        result_csv_path = f'{table_path}/fold_{test_fold}/csv_log/version_{newest_version}/metrics.csv'
        result_csv = pd.read_csv(result_csv_path)
        test_result = result_csv.loc[len(result_csv) - 1]
        test_metrics = [test_result['test/acc'], test_result['test/precision'], 
                        test_result['test/recall'], test_result['test/auroc'], 
                        test_result['test/auprc'], test_result['test/f1'], test_result['test/bacc']]
        metric_df.iloc[test_fold] = test_metrics

    mean_values = metric_df.iloc[0:5].mean()
    std_values = metric_df.iloc[0:5].std()
    new_row = [f"{mean:.4f} ({std:.4f})" for mean, std in zip(mean_values, std_values)]
    metric_df.iloc[5] = new_row
    metric_df.to_csv(f'{table_path}/results.csv')
