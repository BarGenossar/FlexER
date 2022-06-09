# The Battleship Approach to the Low Resource Entity Matching Problem
Code repository for our paper *The Battleship Approach to the Low Resource Entity Matching Problem* (currently under review for [VLDB2023](https://vldb.org/2023/)).


## Abstract
Entity resolution (ER), a longstanding problem of data cleaning and integration, aims at identifying different data records that represent the same real-world entity. Existing approaches treat ER as a universal task, focusing only on finding perfectly matched records and separating the corresponding from non-corresponding ones.

In this work, we use [DITTO](https://github.com/megagonlabs/ditto), the state-of-the-art in universal ER. Specifically, we use the final embedding of the special token [cls](used for classification) as a record pair representation, and inject it into the FlexER system.

![mier_system](/images/mier_system_small.jpg)

## MIER: Multiple Intents Entity Resolution
We offer to extend the universal view of ER, pointedly reflected in the use of a single mapping from records to real-world entities, to include multiple intents.
To better understand the meaning of multiple intents, note that the universal view of ER implicitly assumes a single entity set by which the ER solution must abide.
Such an entity set is not explicitly known, yet typically referred to abstractly as a “real-world entity." We argue that an entity set of choice may vary according to user needs and different users may seek different solutions from the same dataset.


## Methodology
Our proposed solution to the problem of MIER casts the problem as a multi-class multi-label task. As a baseline solution, we first propose to treat multiple intents as a set of independent single intent problems (termed "divide-and-conquer" in our paper), where each intent is considered individually and provides an independent solution for single intent ER (using [DITTO](https://github.com/megagonlabs/ditto)). Additional baseline used in our paper is a simultaneously-learned multiple intents matcher, where labels are predicted cuncurrently.

Our new suggested model, a flexible algorithm to the MIER problem (FlexER), is based on the first baseline by constructing an intents graph using record pairs representations and applying a graph convolutional network (GCN) to provide improved resolutions.
FlexER utilizes the interrelations between intents using the matchers to enrich the ability of the MIER solution as well as the resolution of the individual intents. FlexER
builds upon the solutions of [DITTO](https://github.com/megagonlabs/ditto), by training 𝑃 independent matchers, one for each intent.
Rather than solely depend on the final predictions of the independent matchers, FlexER employs the record pairs representations to construct an intent graph that
is fed into a graph convolutional network (GCN) structure. 
Given an input record pair, an undirected graph is created. The nodes of the graph correspond to the record pair representations for each intent, and the edges to the interrelations among the intents (calculated as the Pearson correlation between the intents' labels over the training set).

The GCN inference is performed using a message passing algorithm following [Kipf & Welling](https://arxiv.org/abs/1609.02907).

For further details and official definitions, please refer to our paper (currently under review).

![FlexER_small](/images/FlexER_small.JPG)

## Requirements
1. The same as mentioned in [DITTO](https://github.com/megagonlabs/ditto)
2. [PyTorch Geometric (Version 1.8.0)](https://pytorch-geometric.readthedocs.io/en/latest/#)

## Getting Started
We provide instructions for the sake of reproducibility.

### Datasets
| Dataset  | # Records | # Pairs | # Intents |
| ------------- | ------------- | ------------- | ------------- |
| AmazonMI  | 3,835  | 15,404  |  5  |
| iTunes-Amazon  | 62,830  | 539  |  4  |
| Walmart-Amazon  | 24,628  | 10,242  |  4  |
| WDC  | 10,935  | 30,673  |  2  |

The used datasets are provided in the [data](./data/) folder, divided to train, validation and test (for each intent).
Explanations about candidate pair representation is provided in [DITTO](https://github.com/megagonlabs/ditto).

Details about the datasets and intents creation are given in our paper (currently under review).


### training with Ditto
To train the independent intent matchers with Ditto (you can apply your own matcher instead), run the following command:
```
python train_ditto.py  --task Amazon/Amazon-Website  \
--batch 32  --max_len 512  --lr 3e-5  --n_epochs 10  \
--finetuning  --save_model  --lm roberta  --intents_num 5
```
The meaning of the flags, excluding intents_num, are described in [DITTO](https://github.com/megagonlabs/ditto).
In order to train the multilabel baseline add "--inference Multilabel"

### Yielding prediction vectors
After training the models for each intent, the creation of pair representations (in our case,  the final embedding of the special token [cls]), can be executed using the following command:
```
python matcher.py  --task Amazon/Amazon-Website  \
--input_path data/Amazon/Amazon-Website  \
--output_path data/Amazon/Amazon-Website  \
--lm roberta  --checkpoint_path checkpoints/  \
--max_len 512 --intents_num 5
```

### Running FlexER
The final stage of our framework is the integration of the independent intent representations. We run it with the following command (for each intent separately):
```
python graph_matcher.py  --task Amazon/Amazon-Website  \
--files_path data/  --n_epochs 10  \
--intent 0  --batch_size 32  \
--intents_num 5  --hidden_channels 1024
```


