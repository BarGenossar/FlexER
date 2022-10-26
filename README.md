
# FlexER: Flexible Entity Resolution for Multiple Intents

FlexER is an approach the multiple intents entity resolution (MIER) problem.
FlexER combines independent intent-based solutions to improve outcome to multiple resolution problems, by using graph neural network (GNN).
The independent intent solutions are based on solutions to universal ER tasks, such that every matcher which can provide a latent representation of record pairs can be adjusted to our framework.
In this work, we use [DITTO](https://github.com/megagonlabs/ditto), the state-of-the-art in universal ER. Specifically, we use the final embedding of the special token [cls](used for classification) as a record pair representation, and inject it into the FlexER system.

The paper was accepted to [SIGMOD2023](https://2023.sigmod.org/).


## MIER: Multiple Intents Entity Resolution
We offer to extend the universal view of ER, pointedly reflected in the use of a single mapping from records to real-world entities, to include multiple intents.
To better understand the meaning of multiple intents, note that the universal view of ER implicitly assumes a single entity set by which the ER solution must abide.
Such an entity set is not explicitly known, yet typically referred to abstractly as a “real-world entity." We argue that an entity set of choice may vary according to user needs and different users may seek different solutions from the same dataset.

A one-size-fits-all resolution provides an adequate solution for universal ER, a standalone task with a single equivalence intent.
Yet, some data cleaning/integration challenges may involve multiple intents. Therefore, instead of performing a universal ER, we argue for enhancing ER to support multiple outcomes for multiple intents. 
A MIER involves a set of (possibly related) entity mappings for a set of intents E = {𝐸1, 𝐸2, · · · , 𝐸𝑃 }, offering multiple ways to divide 𝐷, each serving as a solution for a respective intent.

For further details and official definitions, please refer to our paper (currently under review).

## Requirements
1. The same as [DITTO](https://github.com/megagonlabs/ditto)
2. [PyTorch Geometric (Version 1.6.3)](https://pytorch-geometric.readthedocs.io/en/latest/#)

## Getting Started
We provide instructions for the sake of reproducibility.

### Datasets
| Dataset  | # Records | # Pairs | # Intents |
| ------------- | ------------- | ------------- | ------------- |
| AmazonMI  | 3,835  | 15,404  |  5  |
| Walmart-Amazon  | 24,628  | 10,242  |  4  |
| WDC  | 10,935  | 30,673  |  2  |

The used datasets are provided in the [data](./data/) folder, divided to train, validation and test (for each intent).
Explanations about candidate pair representation is provided in [DITTO](https://github.com/megagonlabs/ditto).

Details about the datasets and intents creation are given in our paper (currently under review).

### training with Ditto
To train the independent intent matchers with Ditto (you can apply your own matcher instead), run the following command:
```
python train_ditto.py  --task Amazon/Amazon-Website  \
--batch 16  --max_len 512  --lr 3e-5  --n_epochs 15  \
--finetuning  --save_model  --lm roberta  --da del  \
--dk product  --summarize  --intents_num 5
```
The meaning of the flags, excluding intents_num, are described in [DITTO](https://github.com/megagonlabs/ditto).

### Creating Prediction Vectors
After training the models for each intent, the creation of pair representations (in our case,  the final embedding of the special token [cls]), can be executed using the following command:
```
python matcher.py  --task Amazon/Amazon-Website  \
--input_path data/Amazon/Amazon-Website  \
--output_path data/Amazon/Amazon-Website  \
--lm roberta  --checkpoint_path checkpoints/  \
--max_len 512 --intents_num 5
```

### Running FlexER
Now we can run FlexER using the following command (for each intent separately):
```
python flexer_main.py  --task Amazon/Amazon-Website  \
--intents_num 5  --k_size 4 \
--generate_neighbors_dict \
--load_datasets \
--files_path data/ \
--batch_size 16  \
--hidden_channels 200 \ 
--epochs_num 150 \
--GNN_model GraphSAGE \
--seeds_num 5
```
Use the flag ''generate_neighbors_dict'' only for the first time you want to create the graph (with the KNN computations). 
If the graph was already created use the ''load_dataset'' flag.
If you want to run an intent ablation study use the ''ablation'' flag.
