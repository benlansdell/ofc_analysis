# OFC decoder

## What it does

Performs a few sanity checks on the data, by seeing if cell tracked over multiple days depend on treatment, or if behavior (e.g. always choosing rewarded side) is biased responsive cell analysis. 

Then compute a mutual information analysis between cell responsive labels between days, showing Idazoxan labels have little relation to previous cell responses. 

Then fit a LDA decoder to the trajectories and show that chlonidine continues to predict reward even after reversal and that idazoxan predictions are more strongly oscillating between one prediction and the other. 

## Setup

This can be run from dnb2's JuptyerHub server: `dnb2:8000`. The default conda environment should have all the packages needed to run these analyses. 

If not, you can install the requirements with
```
pip install -r requirements.txt
```

## How to run

You should run each notebook/script from this base project directory.

### 1 Cell multi-day analysis

Run through `cell_multiday_analysis.ipynb`. The 

### 2 Mutual information 

Run through `mutual_information.ipynb`.

### 3 Decoder analyses

Run `decoder.py`. You can do this from within a terminal with
```
python src/decoder.py
```
Or from within a notebook with: 
```
%run src/decoder.py
```

After this, you can make plots with `decoder_analysis_plots.ipynb`.