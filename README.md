# Interactive_cell_segmentation

## **Setup environment**

To install all necessary packages, run this line in command line:

`pip install -r requirements.txt`

## Configuration file

For running scripts in this project it is necessary to fill configuration file  - `pipeline/config.yaml`.

Paths and directories:
 - *project_dir* - the main directory of the project, all subdirectories should be created in it
 - *train_data_dir* - subdirectory containing training data
 - *test_data_dir* - subdirectory containing test data
 - *train_annotations_dir* - subdirectory containing training annotations
 - *test_annotations_dir* - subdirectory containing test annotations
 - *pos_click_maps_dir* - subdirectory containing positive clicks guidance maps
 - *neg_click_maps_dir* - subdirectory containing negative clicks guidance maps
 - *test_image_path* - path to the test image used as an example in the Interactive Cell Segmentator
 - *models_dir*  - subdirectory with trained models
 - *tensorboard_logs_dir* - subdirectory with logs from Tensorboard callback

Train parameters:
 - *batch_size* - size of the batch
 - *epochs* - maximum number of training epochs
 - *early_stop_patience*  - patience parameter of the early stopping callback
 - *train_val_ratio*  - ratio of the training set to the validation set
 - *buffer_size* - size of the buffer used by the TensorFlow for loading the data
 - *seed* - the seed value

Data parameters:
 - *image_height* - height of the image in pixels
 - *image_width* - width of the image in pixels
 - *image_channels* - number of the image channels
 - *input_channels* - number of the channels in the input tensor for the neural network
 - *pos_click_map_scale* - scale factor of the positive click map
 - *neg_click_map_scale* - scale factor of the negative click map

Model architecture:
 - *shallow_unet* - flag indicating if shallow UNet architecture will be used
 - *fcn* - flag indicating if FCN UNet architecture will be used
 - *attention_dual_path* - flag indicating if shallow EA-DPUNet architecture will be used

GUI parameters:
- *window_size* - size of the GUI's window
- *window_x* - x starting position of the GUI's windows
- *window_y* - y starting position of the GUI's windows
- *img_dpi* - resolution of the GUI's visualizations


## Projects' structure
The project is divided into four main subdirectories:
```
model_architectures/
    attention_module.py         - implementation of the CBAM attention module
    AttentionDualPath.py        - implementation of the EA-DPUNet architecture
    FCN.py                      - implementation of the FCN architecture
    SahallowUnet.py             - implementation of the UNet architecture
models/
    logs/                       - contains logs from TensorBoard callback
    *.hdf5                      - trained models saved in HDF5 format
pipeline/
    config.yaml - configuration files
    generate_click_guidance_maps.py     - script for generation of the guidance maps
    InteractiveCellSegmentator.py       - implementation of the InteractiveCellSegmentator
    loss_and_metrics.py                 - implementations of loss function and training/evaluation metrics
    main.py                             - the main script for running the InteractiveCellSegmentator
    model_evaluate.py                   - script for evalutaing the trained models
    model_train.py                      - script for training the models
utils/
    pyqt_cursor_imgs/       - contains PNG files with positive/negative click cursors
    test_image/             - contains sample test image used in InteractiveCellSegmentator
    configuration.py        - utilisation functions for reading configuration file and training configuration
```


## Projects' directories structure
Below is presented the structure of the projects' directories. Under the *project_dir* parameter from the 
`pipeline/config.yaml` configuration file, there can be any desired directory, but the subdirectories and files
in it have to follow the structure below.
```
D:/path/to/project/dir/
    segmentation_data/
        datasets/
            test/
                image/
                    *.PNG (images of the cells)
                mask/
                    *.PNG (ground truth masks)
                neg_click/
                    *.PNG (negative click guidance maps)
                pos_click/
                    *.PNG (positive click guidance maps)
            train_valid/
                image/
                    *.PNG (images of the cells)
                mask/
                    *.PNG (ground truth masks)
                neg_click/
                    *.PNG (negative click guidance maps)
                pos_click/
                    *.PNG (positive click guidance maps)
        test_annotations/
            *.XML (XML files with annotations)
        train_annotations/
            *.XML (XML files with annotations)
```


## Training and evaluating the models
In order to train the model, the training and validation data must be placed according to the directory structure
shown above. The next step is to set the parameters `batch_size`,` epochs`, `early_stop_patience` and
`train_val_ratio` in the configuration file. Depending on the desired model architecture to be trained, one of the
parameters `shallow_unet`,` fcn`, or `attention_dual_path` must be _ ** True ** _ in the configuration file. After 
setting all the required parameters and correct placement of train data and validation, the training process can be 
started by running the script `pipeline/model_train.py`. The trained model will be saved in the `models/` directory 
in HDF5 format.

In order to evaluate the trained model, analogically to the training process, all the above-mentioned parameters in
configuration file must be set and additionally test data must be placed according to the above directory structure.


## Running InteractiveCellSegmentator
In order to run the InteractiveCellSegmentator tool run script `pipeline/main.py`. For this purpose none of the
parameters in the configuration file has to be changed. The only requirement is that in the firectory `models/`
there has to be trained model file corresponding to the `shallow_unet`,` fcn`, or `attention_dual_path` set in the
configuration file. When you run the script, InteractiveCellSegmentator opens automatically.