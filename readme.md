# The code for InterDDI

## Requirements

All the required packages can be installed by running pip install -r requirements.txt.

```shell
tensorboard==2.9.1
torch==1.11.0+cu113
tqdm==4.61.2
rdkit==2023.9.6
torchdrug==0.2.1
hyperopt==0.2.7
scipy==1.7.3
scikit-learn==1.0.2
pandas
```

## Running scripts

First `cd` into the corresponding directory, i.e., DrugBank , TWOSIDES or Pretrain.

Our data is uploaded on the :

Once the data is ready, you can run the following scripts to reproduce the results:

```shell
python F_train.py 
```

All the hyperms are setting in file.


