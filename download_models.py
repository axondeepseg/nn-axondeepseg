"""
Download packaged nnUNet model 

Author: Armand Collin
"""
from pathlib import Path

MODELS = {
    'model_seg_unmyelinated_tem' : {
        'url': '...',
        'description': 'Unmyelinated axon segmentation (1-class)',
        'contrasts': ['TEM'],
    },
    'model_seg_rabbit_axon-myelin_bf': {
        'url': None,
        'description': 'Axon and myelin segmentation on Toluidine Blue stained BF images (rabbit)',
        'contrasts': ['BF'],
    },
}

def download_models():
    if not (Path('.') / 'models').exists():
        print('No model found. Please select a model to download.')
        model_ids = {}
        number_of_models = len(list(MODELS.keys()))
        for i, model in enumerate(MODELS.keys()):
            model_ids[i] = model
            desc = MODELS[model]['description']
            print(f'[{i}] - {model}:\n\t {desc}')
        model_id = int(input(f'Please select a model ID (from 0 to {number_of_models-1}): '))
        assert model_id >= 0 and model_id < number_of_models, 'Invalid model ID.'
        model = model_ids[model_id]
        print(f'{model} selected. Downloading...')
        

if __name__ == '__main__':
    download_models()