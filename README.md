# Sign Language Recognition Service

This is a Sign Language Recognition service utilizing a deep learning model with Long Short-Term Memory to perform sign language recognition. The service was developed as a part of a bachelor project at Aalborg University. 

## Requirements

- Python 3.7
- OpenPose 1.6.0
- CUDA 10.0
- cuDNN 7.5.0
- Numpy 1.18.5
- OpenCV 4.5.1.48
- Flask 1.1.2
- Tensorflow 2.0.0
- Pandas 1.1.5
- Tensorboard
- Matplotlib
- Seaborn
- Scikit-Learn

## How to use

### Installing OpenPose

1. Please install **OpenPose 1.6.0** for Python by following [the official guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md). Note that the newest release on the OpenPose github is 1.7.0 - for this service to work, 1.6.0 _must_ be used.

    A few things to note when installing OpenPose:
    
    - When cloning the OpenPose repository, use the following git command to get version 1.6.0:
        ```
        git clone --depth 1 --branch v1.6.0 https://github.com/CMU-Perceptual-Computing-Lab/openpose
        ```
    - Remember to run the following command on the newly cloned repository:
        ```
        git submodule update --init --recursive --remote
        ```
    - Use **Visual Studio Enterprise 2017** to build the required files. Install this _first_ if you do not already have it.
    - Install **CUDA 10.0** and **cuDNN 7.5.0** for CUDA 10.0 _after_ installing Visual Studio Enterprise 2017.
    - When generating the files using CMake, make sure that the **BUILD_PYTHON** flag is enabled, and that the Python version is set to **3.7**. Also make sure that the detected CUDA version is **10.0**.
    - After building with Visual Studio Enterprise 2017, make sure that all necessary files have been generated.
      - There should be a **openpose.dll** in _/x64/Release/_
      - There should be a **openpose.exp** and **openpose.lib** in _/src/openpose/Release/_
      - There should be a **pyopenpose.cp37-win_amd64.pyd** in _/python/openpose/Release/_

2. Install requirements from *requirements.txt*
4. Change the path in *main/openpose/paths.py* to the path of your OpenPose installation:
    ```
    # Change this path so it points to your OpenPose path relative to this file
    OPEN_POSE_PATH = get_relative_path(__file__, '../../../../openpose')
    ```
5. If you get any errors related to OpenPose when running the service, please go back and make sure that all instructions have been followed - be particularly careful to install the correct CUDA/cuDNN versions, make sure that the BUILD_PYTHON flag was enabled and that Python 3.7 was used when generating the files.

When OpenPose is successfully installed, you can either use the existing model trained on our dataset, or you can choose to make your own dataset and train a model on this instead.

### Using the service
A singular endpoint *'/recognize'* has been created in order to perform recognition, which allows for POST requests to be made. The endpoint expects a sequence of base64 images, which will get converted into a suitable format recognizable by the classifier. 

### Creating a custom dataset
In order to create a custom dataset, you can access the file **create_dataset.py** and change the following constant:
```
DATASET_NAME = 'dsl_dataset'
```
Such that the path in the constant **DATASET_DIR** points to a folder where the dataset is located. This folder should contain another folder called 'src', which contains folders for all the desired labels in the dataset. Each of these folders should contain videos of the corresponding sign.

Before running the script, the following constants can be tweaked based on the desired settings:

```
WINDOW_LENGTH = 60
STRIDE = 5
BATCH_SIZE = 512
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
```
Finally, the following constant can be changed:
```
CREATE_RAW_DATA = True
```
This is because initial feature extraction by OpenPose can be a fairly lengthy process. This allows for the tweaking of the dataset after features have been extracted, by setting this to **False**. Note that the raw OpenPose data must be created before the actual dataset can be created, so it is necessary to do this at least once.

### Training a custom model

In order to train a custom model you can make use of the **train_models.py** file. Here, the constant **DATASET_NAME** can be changed to reflect the name of the dataset you wish to use, such that the **DATASET_DIR** points to the correct folder. Furthermore, you can specify a tensorboard directory:

```
DATASET_NAME = 'dsl_dataset'
DATASET_DIR = f'.\\main\\algorithm\\datasets\\{DATASET_NAME}'
MODELS_DIR = f'.\\main\\algorithm\\models\\{DATASET_NAME}'
TENSORBOARD_DIR = f'{MODELS_DIR}\\logs'
```

Before running the script, you can tweak various training settings as well as the hyper parameters of the model by changing the following constants: 

```
MODEL_NAME = "model"
EPOCHS = 25
LAYER_SIZES = [64]
DENSE_LAYERS = [0]
DENSE_ACTIVATION = "relu"
LSTM_LAYERS = [2]
LSTM_ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "softmax"
```

Note that the trainer can train multiple models depending on these settings. Changing the **LAYER_SIZES**, **DENSE_LAYERS** and **LSTM_LAYERS** to contain several values will result in a model being trained for each possible combination.

After training your model, you should change the **paths.py** located in *main/core/* to reflect the path to the new model by changing the constant **MODEL_NAME** to the name of your model:

```
MODEL_NAME = 'dsl_lstm.model'
```

Finally, it also possible to generate a confusion matrix for your model by using the **generate_confusion_matrix.py** script. Here, you simply change the constants **DATASET_NAME** and **MODEL_NAME** such that the **DATASET_DIR** points to your dataset directory, and **MODEL_DIR** points to your model directory, respectively:

```
DATASET_NAME = "dsl_dataset"
MODEL_NAME = "dsl_lstm"
DATASET_DIR = f"./main/algorithm/datasets/{DATASET_NAME}/{DATASET_NAME}.pickle"
MODEL_DIR = f"./main/algorithm/models/{DATASET_NAME}/{MODEL_NAME}"
```

Happy signing :O)

## Authors

- Adil Cemalovic
- Martin Lønne
- Magnus Helleshøj Lund