# Classification and spelling models

This folder contains the code used for training classification models and 
simulating the predictions and beam search used in real-time in our paper. 

The notebook `example.ipynb` replicates our main results involving the
decoding of silently attempted speech (silent attempts to "mime" speech)
and using the attempts to spell sentences during real-time sentence spelling. 

To open the notebook, activate the environment and open Jupyter lab in this 
repository.
```bash
$ source activate silent_spelling
$ jupyter lab
```
Then, open the `example.ipynb` notebook and run the cells step by step
to replicate our approach.

## Data
To run the example notebook, you should confirm that the following data are in the appropriate folders.
You may need to move data that you download separately into the appropriate locations.
In the following paths, the initial `.` indicates the path to the `classify_and_spell` folder containing this README file.

Training data for the classification model:
* Neural data during mimed speech: `./data/X_alpha_new_mimed.npy`
* Labels: `./data/Y_alpha_new_mimed.npy`
* Block labels: `./data/blx_alpha_new_mimed.npy`

Real-time spelling data during the copy-typing task: 
* Neural data: `./data/realtime_spelling/block_2726_neural.npy`

Other data that is already included in this repository:
* Date dictionary from block to date: `./data/date_dictionary_mimed.pkl `
* Sentence start times and labels during real-time spelling: `./data/realtime_spelling/block_2726_timing.pkl`

Corpora to train the language model: 
* A corpus of tweets: `./language_models/corpora/en_US.txt`
* A corpus of movie dialogue: `./language_models/corpora/movie_lines.txt`
