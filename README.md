# Learning to Win Lottery Tickets in BERT Transfer via Task-agnostic Mask Training

This repository contains implementation of the [paper](https://openreview.net/forum?id=BRelke4S5l9) "Learning to Win Lottery Tickets in BERT Transfer via Task-agnostic Mask Training" in NAACL 2021.

The code for task-agnostic mask training is based on [huggingface/transformers](https://github.com/huggingface/transformers) and [maskbert](https://github.com/ptlmasking/maskbert).

The code for downstream fine-tuning and IMP is modified from [BERT-Tickets](https://github.com/VITA-Group/BERT-Tickets).


## Overview

### Method: Task-Agnostic Mask Training (TAMT)

TAMT learns the subnetwork structures on the pre-training dataset, using either the MLM loss or the KD loss. The identified subnetwork is then fine-tuned on a range of downstream tasks, in place of the original BERT model.

![](./figures/method.png)

### Pre-training and Downstream Performance

The pre-training performance of a BERT subnetwork correlates with its down-stream transferability.

![](./figures/loss_acc.PNG)



## Requirements

Python3 <br />
torch>1.4.0 <br />


## Pruning and Mask Training

### TAMT-MLM
