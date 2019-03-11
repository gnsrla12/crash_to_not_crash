## Crash to Not Crash: Learn to Identify Dangerous Vehicles using a Simulator
### [Tensorflow](https://github.com/gnsrla12/GTACrash) | [project page](https://sites.google.com/view/crash-to-not-crash) |   [paper](http://csuh.kaist.ac.kr/Suh_Crash_AAAI.pdf)


## Getting Started
### Installation
- Install tensorflow
- Clone this repo:
```bash
git clone https://github.com/gnsrla12/GTACrash
cd gtacrash_for_distrib
```

### Apply a Pre-trained Model
- Download the GTACrash dataset and YouTubeCrash dataset:
```
bash ./datasets/download_dataset.sh ae_photos
```
- Download the pre-trained model `style_cezanne` (For CPU model, use `style_cezanne_cpu`):
```
bash ./pretrained_models/download_model.sh style_cezanne
```
- Now, let's measure performance of our model on the YouTube test dataset:
```
python ./script/test.py
```
The test results will be printed to `./results/style_cezanne_pretrained/latest_test/index.html`.  
Please refer to [Model Zoo](#model-zoo) for more pre-trained models.
`./examples/test_vangogh_style_on_ae_photos.sh` is an example script that downloads the pretrained Van Gogh style network and runs it on Efros's photos.

### Train
- Download a dataset (e.g. zebra and horse images from ImageNet):
```bash
bash ./datasets/download_dataset.sh horse2zebra
```
- Train a model:
```bash
DATA_ROOT=./datasets/horse2zebra name=horse2zebra_model th train.lua
```
- (CPU only) The same training command without using a GPU or CUDNN. Setting the environment variables ```gpu=0 cudnn=0``` forces CPU only
```bash
DATA_ROOT=./datasets/horse2zebra name=horse2zebra_model gpu=0 cudnn=0 th train.lua
```
- (Optionally) start the display server to view results as the model trains. (See [Display UI](#display-ui) for more details):
```bash
th -ldisplay.start 8000 0.0.0.0
```

### Test
- Finally, test the model:
```bash
DATA_ROOT=./datasets/horse2zebra name=horse2zebra_model phase=test th test.lua
```
The test results will be saved to an HTML file here: `./results/horse2zebra_model/latest_test/index.html`.

