"""
Run inference on 2D images using pre-trained nnUNetv2 model.
The output can either be 1 class (unmyelinated axons only) or 
2-class (axon + myelin).

Authors: Armand Collin, Naga Karthik
"""

import os
import argparse
import torch
import cv2
import tempfile
import shutil
from tqdm import tqdm
from loguru import logger
from pathlib import Path

import download_models

# setup dummy env variables so that nnUNet does not complain
os.environ['nnUNet_raw'] = 'UNDEFINED'
os.environ['nnUNet_results'] = 'UNDEFINED'
os.environ['nnUNet_preprocessed'] = 'UNDEFINED'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment images using nnUNet')
    parser.add_argument('--seg-type', type=str, required=True,
                        help='Segmentation type. Use AM for axon and myelin segmentation (2-class) or UM for unmyelinated axons (1-class)')
    parser.add_argument('--path-dataset', default=None, type=str,
                        help='Path to the test dataset folder. Use this argument only if you want to predict on a whole dataset.')
    parser.add_argument('--path-images', default=None, nargs='+', type=str,
                        help='List of images to segment. Use this argument only if you want to predict on a single image or list of invidiual images.')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--path-model', default=None,
                        help='Path to the model directory. This folder should contain individual folders like fold_0, fold_1, etc.',)
    parser.add_argument('--use-best', action='store_true', default=False,
                        help='Use the best checkpoints instead of the final ones. Default: False')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False')
    parser.add_argument('--fold-all', action='store_true', default=False,)
    return parser

def rescale_predictions(outpath, segtype):
    predictions = Path(outpath).glob('*.png')
    rescaling_factor = 255
    if segtype == 'AM':
        rescaling_factor = 127
    
    for pred in tqdm(predictions):
        img = cv2.imread(str(pred), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(pred), img*rescaling_factor)

def delete_dir(path_dir):
    for f in path_dir.glob('*'):
        f.unlink()
    path_dir.rmdir()

def main():
    parser = get_parser()
    args = parser.parse_args()

    # find available models
    models = download_models.get_downloaded_models()
    if args.path_model == None:
        if len(models) == 0:
            logger.error('No model downloaded. Run the download_models.py script first.')
            return 1
        elif len(models) == 1:
            path_model = models[0]
            logger.info(f'A single model was found: {path_model}. It will be used by default.')
        elif len(models) > 1:
            logger.error('Multiple models were found in the models/ folder. Please use the --path-model argument to disambiguate.')
            return 1
    else:
        path_model = args.path_model

    assert args.seg_type == 'AM' or args.seg_type == 'UM', 'Please select a valid segmentation type.'

    if args.path_dataset is not None and args.path_images is not None:
        raise ValueError('You can only specify either --path-dataset or --path-images (not both). See --help for more info.')

     # find all available folds in the model folder
    if args.fold_all:
        folds_avail = ['all']
    else:
        folds_avail = [int(str(f).split('_')[-1]) for f in Path(args.path_model).glob('fold_*')]

    
    # instantiate nnUNetPredictor
    predictor = nnUNetPredictor(
        perform_everything_on_gpu=True if args.use_gpu else False,
        device=torch.device('cuda', 0) if args.use_gpu else torch.device('cpu'),
    )
    logger.info('Running inference on device: {}'.format(predictor.device))
    # initialize network architecture, load checkpoint
    checkpoint_name = 'checkpoint_final.pth' if not args.use_best else 'checkpoint_best.pth'
    predictor.initialize_from_trained_model_folder(
        path_model, 
        use_folds=folds_avail,
        checkpoint_name=checkpoint_name
    )
    logger.info('Model loaded successfully.')


    if args.path_dataset is not None:
        datapath = Path(args.path_dataset)
        assert datapath.exists(), 'The specified path-dataset does not exist.'

        logger.info('Creating temporary input directory.')
        tmp_dir = Path('.') / 'tmp'
        try:
            tmp_dir.mkdir()
        except FileExistsError:
            delete_dir(tmp_dir)
            tmp_dir.mkdir()
        for fname in Path(args.path_dataset).glob('*.png'):
            target_fname = f'{fname.stem}_0000{fname.suffix}'
            shutil.copyfile(str(fname), tmp_dir / target_fname)
        predictor.predict_from_files(str(tmp_dir), args.path_out)
        logger.info('Rescaling predictions to 8-bit range.')
        rescale_predictions(args.path_out, args.seg_type)

        logger.info('Deleting temporary directory')
        delete_dir(tmp_dir)

    elif args.path_images is not None:
        logger.warning('path-images not yet supported')


if __name__ == '__main__':
    main()
