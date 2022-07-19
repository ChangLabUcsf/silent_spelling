# Generating manuscript figures

This folder contains the code to generate the figures as they appear in the manuscript and supplement.

To open the notebooks, activate the environment and open Jupyter lab in this
repository.
```bash
$ source activate silent_spelling
$ jupyter lab
```

## Data

All notebooks work off of the `source_data.xlsx` file that is at the following path.
```
/path/to/silent_spelling/source_data/source_data.xlsx
```
Here, the `/path/to` part should be replaced with the path to the top-level of this repository.


## Folder organization and other files

1. `figures/saved_figures`: A `saved_figures` folder will need to be created within the `figures` folder. PDFs and PNGs of the generated figures can be saved there.
2. `figures/recon`: This contains files related to the MRI brain reconstruction and electrode coordinates for our participant.
3. `silent_spelling/silent_spelling/*.py`: This folder contains all plotting and utility functions used to create the figures.
4. `silent_spelling/letter_codeword_label_mapping.pkl`: This file is a dictionary corresponding between the numerical labels for the English alphabet and NATO phonetic alphabet.

## Fonts
If there are issues loading the fonts (e.g. `findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.`)
Then you will need to first conda install the `mscorefonts` package and clear the `matplotlib` cache folder in your home directory.
```bash
$ source activate silent_spelling
$ conda install -c conda-forge mscorefonts
$ rm ~/.cache/matplotlib -rf
```

