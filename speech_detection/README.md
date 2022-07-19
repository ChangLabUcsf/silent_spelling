# The speech detection model

This folder contains the code to train a speech detection model with the same
model configuration as used in the paper.

The `train_speech_detection_model.ipynb` contains the code to train a model
with example data (if you have it), as well as how to load a trained model and predict on an example block of real-time spelling.
The pretrained model used in the paper is also provided.

To open the notebook, activate the environment and open Jupyter lab in this
repository.
```bash
$ source activate silent_spelling
$ jupyter lab
```
Then, open the `train_speech_detection_model.ipynb` notebook and run the
cells step by step to replicate our training approach.

## Data

If you have the de-identified speech detection data, place it in a folder named `data`
at the following path:

```
/path/to/silent_spelling/speech_detection/bravo1/data
```

Here, the `/path/to` part should be replaced with the path to the top-level of this repository.
This new `data` directory should exist alongisde a file named `demo_block_splits.pkl`. 
These `.h5` files contain the neural data (electrocorticography, ECoG) for example blocks of silent speech attempts and attempted hand movements and the labels for `silence`, `speech`, `speech_preparation`, and `motor`, as defined in the Supplement of the paper.

There is also a `trial_mapping.h5` file which defines indices for 0.5 s long chunks of data.
This mapping is used to pull trials into memory only before they are used for a batch update during training.

256 neural features are included in the `.h5` files. 
The first 128 correspond to the high-gamma activity for each of the 128 electrodes in the ECoG array, and the second 128 correspond to the low-frequency signals for each of the 128 electrodes in the ECoG array.
Both feature types are sampled at 200 Hz.

## Folder organization and other files

1. `pretrained_example_model`: This contains a pretrained model and the model
   configuration JSON used to restore other parts of the model.
2. `bravo1/data`: This contains the labeled training data.
3. `bravo1/results`: This contains 2 model folders, one with a prefix of
   `temporary_`. The temporary folder is used to save the model on each
   training loop, while the other is used to save the best model.
4. `speech_detection/*.py`: These files contains custom code for training and
   testing the speech detection model and are referenced by the notebook.
5. `speech_detection/bravo1/demo_block_splits.pkl`: This contains the training
   splits for each utterance set used. The model uses this to identify which
   blocks should be loaded and used for training.


