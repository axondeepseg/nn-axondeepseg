"""
Download packaged nnUNet model 

Author: Armand Collin
"""
from pathlib import Path
import shutil
import tempfile
import cgi
import zipfile
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry

MODELS = {
    'model_seg_unmyelinated_tem' : {
        'url': 'https://github.com/axondeepseg/model_seg_unmyelinated_tem/releases/download/v1.0.0/model_seg_unmyelinated_tem.zip',
        'description': 'Unmyelinated axon segmentation (1-class)',
        'contrasts': ['TEM'],
    },
    'model_seg_rabbit_axon-myelin_bf': {
        'url': None,
        'description': 'Axon and myelin segmentation on Toluidine Blue stained BF images (rabbit)',
        'contrasts': ['BF'],
    },
}


def get_downloaded_models():
    return list(Path('models').glob('*'))

def download_data(url_data):
    """ Downloads and extracts zip files from the web. Taken from AxonDeepSeg.ads_utils
    :return: 0 - Success, 1 - Encountered an exception.
    """
    # Download
    try:
        print('Trying URL: %s' % url_data)
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=retry))
        response = session.get(url_data, stream=True)
        print(response)

        if "Content-Disposition" in response.headers:
            _, content = cgi.parse_header(response.headers['Content-Disposition'])
            zip_filename = content["filename"]
        else:
            print("Unexpected: link doesn't provide a filename")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / zip_filename
            with open(tmp_path, 'wb') as tmp_file:
                total = int(response.headers.get('content-length', 1))
                tqdm_bar = tqdm(total=total, unit='B', unit_scale=True, desc="Downloading", ascii=True)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        dl_chunk = len(chunk)
                        tqdm_bar.update(dl_chunk)
                tqdm_bar.close()
            # Unzip
            print("Unzip...")
            try:
                with zipfile.ZipFile(str(tmp_path)) as zf:
                    zf.extractall(".")
            except (zipfile.BadZipfile):
                print('ERROR: ZIP package corrupted. Please try downloading again.')
                return 1
            print("--> Folder created: " + str(Path.cwd() / Path(zip_filename).stem))
    except Exception as e:
        print("ERROR: %s" % e)
        return 1
    return 0

def download_models():
    if len(get_downloaded_models()) == 0:
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
    url = MODELS[model]['url']
    print(f'{model} selected. Downloading from {url}')
    (Path('.') / 'models').mkdir(exist_ok=True)

    if download_data(url) == 0:
        print('Model successfully downloaded.')
        shutil.move(model, f'models/{model}')

if __name__ == '__main__':
    download_models()