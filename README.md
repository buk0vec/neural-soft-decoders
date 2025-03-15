# Soft Neural Decoders

## EE 387 Final Project W25

Hi! This is the code for my final project, titled **Constructing and Evaluating Neural Soft Decoders for Binary Linear Codes**. Feel free to poke around.

## Usage

Oh wait, you actually want to run this? 

1. `pip install -r requirements.txt`
2. To train a model locally, run `python3 experiment.py --model [rnn|gru|transformer] --code BCH_N63_K45` (you may have to modify the script to support your device)
3. To train a model remotely, first set up Modal with `modal setup` before running `modal run experiment_remote.py --model [rnn|gru|transformer] --code BCH_N63_K45`. The model will be saved to a new volume.
4. To evaluate one or more of the models in the `models` folder, run `python3 eval.py --model experiment_BCH_N63_K36_gru_lr=0.0001_epochs=200 experiment_BCH_N63_K36_rnn_lr=0.0001_epochs=200`.