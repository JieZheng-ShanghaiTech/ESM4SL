# ESM4SL

This is the code of **ESM4SL**, which is a novel method using the pre-trained protein language model ESM-2 for the prediction of cancer cell-line-specific synthetic lethality.

## Usage 

### Environment
First, you can create the environment required for running our codes by
```
bash environment.sh
```

### Preprocess

All the codes for data preprocessing is in `data_preprocess/`.

### Main
A simple example of running our program is
```
bash script/attn/train.sh
```
After the program finishes, you can see the outputs in `output/attn/` by
```
tensorboard --logdir output/attn/<name_of_your_run>
```

If you would like to run multiple cell lines or scenes in one script, you can refer to `script/attn/specific_all.py`.

If you would like to run **ESM-2+MLP** instead of **ESM4SL**, you can refer to the same files in `output/mlp/`.

## References

Our codes is based on [coach-pl](https://github.com/DuskNgai/coach-pl). We thank the authors for their great foundational work.

Some of our codes are credited to [ESM](https://github.com/facebookresearch/esm).
