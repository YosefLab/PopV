# POPV

Create a new anaconda environment with:
```python
conda create -n yourenv python=3.7
conda activate yourenv
conda install -c anaconda -c conda-forge cudnn==8.2.1.32 cudatoolkit==11.0.221
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install popv
```
