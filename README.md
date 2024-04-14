# Fake Video Classifier
## Installation

To build image inside the directory, run following commands

```sh
docker build -t 3d_image .
docker run -p 8888:8888 3d_image
```
This docker container will launch a jupyter notebook. In terminal you will see a link which starts with http://127.0.0.1:8888/tree?token= . 
## File structure

| File | README |
| ------ | ------ |
| main.py | Contains the actual code that we used to train the network|
| model.py | Contains the Neural Network class |
| data_loader.py| Custom PyTorch data loader class |
| utils.py | Functions which processes videos into frames to pass through network |
| fake_class2_29_checkpoint.pth.tar | The weights of the network that we recorded after experimentations |
| demo_data/* | We included some videos to make demo |