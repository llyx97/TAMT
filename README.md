# Learning to Win Lottery Tickets in BERT Transfer via Task-agnostic Mask Training

This repository contains implementation of the [paper](https://openreview.net/forum?id=BRelke4S5l9) "Learning to Win Lottery Tickets in BERT Transfer via Task-agnostic Mask Training" in NAACL 2021.

The code for task-agnostic mask training is based on [huggingface/transformers](https://github.com/huggingface/transformers) and [maskbert](https://github.com/ptlmasking/maskbert).

The code for downstream fine-tuning and IMP is modified from [BERT-Tickets](https://github.com/VITA-Group/BERT-Tickets).

## Overview

### Method: Task-Agnostic Mask Training (TAMT)

![](./figures/method.png)

### Correlation between the pre-training and downstream performance of BERT subentworks

![](./figures/loss_acc.PNG)
