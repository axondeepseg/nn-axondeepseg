# nnAxonDeepSeg
1-class and 2-class segmentation of axon/myelin using nnunetv2 

### Usage
Inside a virtual environment (pipenv, conda, etc.), install the requirements:
```
pip install -r requirements.txt
```

The inference tool should now be ready to use. First, download a model using 
```
python download_model.py
```

Then, you can use the `nn_axondeepseg.py` script to apply the model to your images. 
