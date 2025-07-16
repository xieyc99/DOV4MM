### 1. Download ImageNet-1K pre-trained model from the official repository or MMSelfSup. (This model is considered a suspicious model $M_{sus}$.)
MAE: https://github.com/facebookresearch/mae.

CAE: https://github.com/lxtGH/CAE.

iBOT: https://github.com/bytedance/ibot.

SimMIM: https://github.com/microsoft/SimMIM.

Other Models: https://mmselfsup.readthedocs.io/en/latest/model_zoo.html.

### 2. Train the reconstructor $M_r$.
```
python train_reconstructor.py
```

### 3. Compute $\Delta \mathcal{R}_{vt}$ and $\Delta \mathcal{R}_{pt}$, then perform hypothesis testing.
```
python test_reconstructor.py
```