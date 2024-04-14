# Fake Video Classifier
## Installation

To build image inside the directory, run following commands. This might take some time due to dependency download. Especially PyTorch takes significant amount of time.

```sh
docker build -t 3d_image .
docker run -p 8888:8888 3d_image
```
This docker container will launch a jupyter notebook. In terminal you will see a link which starts with
```sh
http://127.0.0.1:8888/tree?token=${TOKEN}
```
Inside the jupyter notebook, demo.ipynb contains necessary instructions for demo. **Note that our actual experimentations has been done in Google Colab.**
## File structure

| File | README |
| ------ | ------ |
| main.py | Contains the actual code that we used to train the network.|
| model.py | Contains the Neural Network class. |
| data_loader.py| Custom PyTorch data loader class to properly load data into network.|
| utils.py | Functions which processes videos into frames to pass through network as sequence of images. |
| fake_class2_29_checkpoint.pth.tar | The weights of the network that we recorded after several experimentations with different configurations.|
| demo_data/* | We included some videos and tensors in case github clone fails. |
| 3d_reconstruction_data/* | To make submission file smaller, docker clones portion of data from the personal git repository. Otherwise, submission takes around 60-70mb.  |

Please refer the actual pdf for our reasonings and results.