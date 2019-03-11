## Crash to Not Crash: Learn to Identify Dangerous Vehicles using a Simulator
### [Tensorflow](https://github.com/gnsrla12/CrashToNotCrash_code) | [Project page](https://sites.google.com/view/crash-to-not-crash) |   [Paper](http://csuh.kaist.ac.kr/Suh_Crash_AAAI.pdf)


## Getting Started
### Installation
- Install tensorflow
- Clone this repo:
```bash
git clone https://github.com/gnsrla12/GTACrash
cd gtacrash_for_distrib
```

### Apply a Pre-trained Model
- Download YouTubeCrash (Real accident dataset collected from YouTube):
```
bash ./datasets/download_dataset.sh ae_photos
```
- Download the pre-trained model trained on GTACrash dataset:
```
bash ./pretrained_models/download_model.sh style_cezanne
```
- Now, let's measure performance of our model on the YouTube test dataset:
```
python ./scripts/test_script.py
```
The test results will be printed.  

### Train
- Download GTACrash (Synthetic accident dataset collected from Grand Theft Auto V):
```bash
bash ./datasets/download_dataset.sh horse2zebra
```
- Train a model on the GTACrash dataset:
```bash
python ./scripts/train_gta_script.py
```

- Train a model on the YouTubeCrash dataset:
```bash
python ./scripts/train_yt_script.py
```

Models will be saved in the folder `./checkpoints/`.

### Visualize
- Finally, visualize the prediction results of the model:
```bash
python ./scripts/visualize_script.py
```
The visualized results will be saved to : `./visualization/`.

