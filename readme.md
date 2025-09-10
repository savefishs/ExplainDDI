# A Knowledge Guided Self-supervised Molecular Visual Model for Drug Interaction Understanding (ExplainDDI)
This is the code necessary to run experiments on the ExplainDDI algorithm.

## Abstract
Understanding drug–drug interactions (DDIs), particularly those involving novel or under-characterized drugs, presents a persistent challenge in pharmacology and drug safety assessment. Existing models typically rely on labeled interaction data and struggle to generalize beyond known drug pairs, while offering limited explanability into the biological mechanisms underlying predicted interactions. We propose ExplainDDI, a knowledge guided molecular visual foundation model for drug interaction understanding, which integrates self-supervised representation learning from molecular structures with semantic reasoning over biomedical knowledge graphs. This unified design enables the model to learn transferable, chemically grounded embeddings even for drugs with no prior interaction annotations. Crucially, ExplainDDI provides multi-level interpretability: at the molecular level, it highlights interaction-relevant metabolic substructures, and at the knowledge level, it identifies semantically coherent relational paths, formed by shared targets, metabolic enzymes, or *gene–disease–drug* cascades within biomedical graphs. Across multiple datasets, including inductive and long-tail DDI prediction tasks, ExplainDDI achieves superior faithfulness while offering mechanistically meaningful explanations. These results demonstrate the potential of combining visual foundation modeling with structured knowledge to enable generalizable, faithful, and interpretable pharmacological inference.

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

Our data is uploaded on the [google drive link](https://drive.google.com/file/d/1qH_hmJB3EVFSuuOlALAaekQUqxCFiFwm/view?usp=drive_link)

Once the data is ready, you can run the following scripts to reproduce the results:

```shell
python {dataset}/F_train.py 
```

All the hyperms are setting in file.

## Acknowledge
The code is implemented based on MAE_pytorch (https://github.com/IcarusWizard/MAE/tree/main). The benchmark datasets are from [EmerGNN](https://github.com/LARS-research/EmerGNN) (DrugBank & TWOSIDES & inductive), and process few-shot data based on [META-DDIE](https://github.com/YifanDengWHU/META-DDIE).
We thank them very much.
