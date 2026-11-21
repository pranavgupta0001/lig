# Laser-Induced Graphene (LIG) Machine Learning

This project provides a small, reproducible Python pipeline for predicting
laser parameters and LIG properties from polymer precursor structures.

Polymer SMILES strings can be converted into:

- molecular fingerprints,
- RDKit-style molecular descriptors, and
- tokenized representations.

The model factory supports random forest, gradient boosting, fully connected
neural networks, and XGBoost (when the corresponding optional packages are
installed).

## Installation

The core representation helpers work without third-party dependencies.
Install the optional modeling dependencies with:

```bash
pip install -e ".[ml]"
```

Use `[rdkit]` for RDKit descriptor support or `[bert]` for pretrained BERT
tokenization:

```bash
pip install -e ".[rdkit,ml]"
pip install -e ".[bert]"
```

## Example

```python
from lig import train_and_evaluate

smiles = ["*CC*", "*CCO*", "*C=C*", "*CCC*", "*COC*", "*CS*"]
laser_power = [10.0, 12.0, 11.0, 9.0, 14.0, 15.0]

model, report = train_and_evaluate(
    smiles,
    laser_power,
    representation="fingerprint",
    model="random_forest",
)
print(report)
```
