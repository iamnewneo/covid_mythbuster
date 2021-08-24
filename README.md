## General Info
This repository includes source code for the COVID Scientific Claim Validation project. The main idea behind the project is to predict whether the current claim is SUPPORTED or REJECTED by the current scientific publications. We achieve this using a twin RoBERTa model. The model also gives the relevant sentences from the abstract of peer-reviewed papers that support or reject the claim.

## Code Structure
1. Most of the training code is inside ./covid_mythbuster/train/
2. Dataset classes are in ./covid_mythbuster/data_loader/
3. Model classes are in ./covid_mythbuster/data_loader/

## Requirements
Python3, Pytorch, transformers, tqdm

## Setup
To run this project install the requirements and make sure you have at least 14GB GPU.
```
bash train.sh
```

## Todo
1. Add more comments
