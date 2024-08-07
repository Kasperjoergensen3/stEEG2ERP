# stEEG2ERP: Learning the Event-Related Potential from Single-Trial Electroencephalography Signals
The code is build on top of the code base given in the [CSLP-AE repository](https://github.com/andersxa/CSLP-AE/blob/main/readme.md).


## Installation
Required packages and specific versions used in the paper are given in `requirements.txt`, and can be installed using pip:

```bash
pip install -r requirements.txt
```
## Usage
First thing is data preparation. Data must be downloaded and the folder structure kept intact. The data can be downloaded from [the ERP Core repository](https://doi.org/10.18115/D5JW4R) and inserted into the `data_preparation/raw`folder. In the `data_preparation` folder the `create_dataset.py` file will create a Pickle file containing all examples, subject labels and task labels for the dataset saved in the `data_preparation/processed` folder. The file can be run with the following command:
```bash
python data_preparation/create_dataset.py
```
Afterwards we need to call
```bash
python data_preparation/create_bootstrap_targets.py
```
which creates a file `data_preparation/processed/targets_bootstrap_200.pt` containing a dictionary mapping subject i, task j as "i,j" to a tensor with 200 bootstrapped ERP targets.

## Reproduced Training Results
command for training model with including all loses and using bootstrap:
```bash
python train.py --recon_enabled 1 --sub_contra_s_enabled 1 --task_contra_t_enabled 1 --restored_permute_s_enabled 1 --restored_permute_t_enabled 1 --epochs 200 --bootstrap 1
```
trained models will be stored in `trained_models`folder. where they can be accesed from evaluation scripts.
## Reproducing Evaluation Results
i will insert the missing code when i have cleaned it up a bit.
