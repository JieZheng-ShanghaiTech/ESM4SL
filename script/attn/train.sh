CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --config-file esm4sl/configuration/attn/new.yaml \
    --num-gpus 1 \
    OUTPUT_DIR /sharedata/home/daihzh/protein/ESM4SL/output/attn/HS944T/C1/fold_0 \
    DATASET.TRAIN_FILE /sharedata/home/daihzh/protein/ESM4SL/data/SLKB/HS944T/C1/sl_train_0.csv \
    DATASET.VAL_FILE /sharedata/home/daihzh/protein/ESM4SL/data/SLKB/HS944T/C1/sl_val_0.csv \
    DATASET.TEST_FILE /sharedata/home/daihzh/protein/ESM4SL/data/SLKB/HS944T/C1/sl_test_0.csv