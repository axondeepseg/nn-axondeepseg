# nnAxonDeepSeg
1-class and 2-class segmentation of axon/myelin using nnunetv2 

### Usage
Inside a virtual environment (pipenv, conda, etc.), install the requirements:
```
pip install -r requirements.txt
```

The inference tool should now be ready to use. First, download a model using the following command. The user will be prompted to specify which model to download.
```
python download_model.py
```

Then, you can use the `nn_axondeepseg.py` script to apply the model to your images. Assuming the images are in a folder called `input`, you can use 
```
python nn_axondeepseg.py --seg-type UM --path-out output-folder --path-dataset input
```
The `--seg-type` argument is used to specify which kind of model is used: *UM* stands for unmyelinated axon, for which we expect a single class output; *AM* stands for axon and myelin, for which we expect a 2-class output. The user can specify any nnUNet model using the `--path-model` argument.
