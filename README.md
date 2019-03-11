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
- Download the YouTubeCrash dataset:
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
The test results will be printed.  

### Train
- Download the GTACrash dataset (Synthetic accident data collected from Grand Theft Auto V):
```bash
bash ./datasets/download_dataset.sh horse2zebra
```
- Train a model:
```bash
DATA_ROOT=./datasets/horse2zebra name=horse2zebra_model th train.lua
```

### Test
- Finally, test the model:
```bash
DATA_ROOT=./datasets/horse2zebra name=horse2zebra_model phase=test th test.lua
```
The test results will be saved to an HTML file here: `./results/horse2zebra_model/latest_test/index.html`.

