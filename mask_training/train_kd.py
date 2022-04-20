# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json, math

import numpy as np
import torch
from collections import namedtuple
from tempfile import TemporaryDirectory
from pathlib import Path
import utils.param_parser as param_parser
import masking.sparsity_control as sp_control
import masking.maskers as maskers
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import MSELoss
import torch.nn.utils.prune as prune

from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling import TinyBertForPreTraining, BertModel, RobertaForPreTraining, RobertaModel
from transformer.tokenization import BertTokenizer
from hg_transformers import RobertaTokenizer
from transformer.optimization import BertAdam

csv.field_size_limit(sys.maxsize)


try:
    from tensorboardX import SummaryWriter
    _has_tensorboard = True
except ImportError:
    _has_tensorboard = False
def is_tensorboard_available():
    return _has_tensorboard

# This is used for running on Huawei Cloud.
oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class CosineLoss(torch.nn.Module):
    def forward(self, student_rep, teacher_rep):
        return (1-torch.cosine_similarity(student_rep, teacher_rep, dim=-1)).mean()


def see_weight_rate(mask, model_type):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(mask['%s.encoder.layer.%d.attention.self.query.weight'%(model_type, ii)].nelement())
        zero_sum = zero_sum+float(torch.sum(mask['%s.encoder.layer.%d.attention.self.query.weight'%(model_type, ii)] == 0))

        sum_list = sum_list+float(mask['%s.encoder.layer.%d.attention.self.key.weight'%(model_type, ii)].nelement())
        zero_sum = zero_sum+float(torch.sum(mask['%s.encoder.layer.%d.attention.self.key.weight'%(model_type, ii)] == 0))

        sum_list = sum_list+float(mask['%s.encoder.layer.%d.attention.self.value.weight'%(model_type, ii)].nelement())
        zero_sum = zero_sum+float(torch.sum(mask['%s.encoder.layer.%d.attention.self.value.weight'%(model_type, ii)] == 0))

        sum_list = sum_list+float(mask['%s.encoder.layer.%d.attention.output.dense.weight'%(model_type, ii)].nelement())
        zero_sum = zero_sum+float(torch.sum(mask['%s.encoder.layer.%d.attention.output.dense.weight'%(model_type, ii)] == 0))

        sum_list = sum_list+float(mask['%s.encoder.layer.%d.intermediate.dense.weight'%(model_type, ii)].nelement())
        zero_sum = zero_sum+float(torch.sum(mask['%s.encoder.layer.%d.intermediate.dense.weight'%(model_type, ii)] == 0))

        sum_list = sum_list+float(mask['%s.encoder.layer.%d.output.dense.weight'%(model_type, ii)].nelement())
        zero_sum = zero_sum+float(torch.sum(mask['%s.encoder.layer.%d.output.dense.weight'%(model_type, ii)] == 0))


    sum_list = sum_list+float(mask['%s.pooler.dense.weight'%model_type].nelement())
    sum_list = sum_list+float(mask['%s.embeddings.word_embeddings.weight'%model_type].nelement())
    zero_sum = zero_sum+float(torch.sum(mask['%s.pooler.dense.weight'%model_type] == 0))
    zero_sum = zero_sum+float(torch.sum(mask['%s.embeddings.word_embeddings.weight'%model_type] == 0))
    return 100*zero_sum/sum_list


def pruning_model_with_mask(model, mask_dict):
    parameters_to_prune =[]
    mask_list = []
    for ii in range(12):
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.query)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query.weight'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.key)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key.weight'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.value)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value.weight'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.output.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense.weight'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].intermediate.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense.weight'])
        parameters_to_prune.append(model.bert.encoder.layer[ii].output.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.output.dense.weight'])

    parameters_to_prune.append(model.bert.pooler.dense)
    mask_list.append(mask_dict['bert.pooler.dense.weight'])

    for ii in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list[ii])
    modules_to_prune = [key.replace('.weight', '') for key in mask_dict.keys()]
    for name, module in model.named_modules():
        if name in modules_to_prune:
            prune.remove(module, 'weight')

#def load_mask(args):
#    mask_seed = args.mask_seed if 'rand' in args.prun_type else ''
#    mask_dir = os.path.join(args.root_dir, 'models/prun_bert',
#            args.prun_type, str(args.zero_rate), str(mask_seed), 'mask.pt')
#    mask = torch.load(mask_dir)
#    return mask

def save_model_mask(model, output_dir, model_type, is_save=True):
    mask_dict = {}
    logger.info('Collecting mask...')
    for name, module in model.named_modules():
        if hasattr(module, 'threshold'):
            mask = module.weight_mask
            mask = binarizer_fn1(mask, module.threshold).bool().cpu()
            mask_dict[name+'.weight'] = mask

    zero = see_weight_rate(mask_dict, model_type)
    logger.info("zero rate: %.5f"%zero)

    if is_save:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model mask to %s", output_dir)
        torch.save(mask_dict, os.path.join(output_dir, 'mask.pt'))
    return zero


class _Binarizer2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, num_to_mask, num_heads):
        return binarizer_fn2(inputs, num_to_mask, num_heads)

    @staticmethod
    def backward(ctx, gradOutput):
        return (gradOutput, None, None)

def binarizer_fn2(inputs, num_to_mask, num_heads):
    outputs = inputs.clone()
    heads_to_mask = outputs.view(-1).sort()[1]
    outputs[:,:] = 1.0
    for i, head in enumerate(heads_to_mask):
        if i==num_to_mask:
            break
        layer_idx = head.item() // num_heads
        head_idx = head.item() % num_heads
        outputs[layer_idx][head_idx] = 0.0
    return outputs

class _Binarizer1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        return binarizer_fn1(inputs, threshold)

    @staticmethod
    def backward(ctx, gradOutput):
        return (gradOutput, None, None)

def binarizer_fn1(inputs, threshold):
    outputs = inputs.clone()
    outputs[inputs.le(threshold)] = 0.0
    outputs[inputs.gt(threshold)] = 1.0
    return outputs

def init_masker(conf, model, logger):
    # init the masker scheduler.

    conf.masking_scheduler_conf_ = (
        param_parser.dict_parser(conf.masking_scheduler_conf)
        if conf.masking_scheduler_conf is not None
        else None
    )
    conf.masking_scheduler_conf_['final_sparsity'] = conf.zero_rate
    if conf.masking_scheduler_conf is not None:
        for k, v in conf.masking_scheduler_conf_.items():
            setattr(conf, f"masking_scheduler_{k}", v)
    conf.logger = logger

    masker_scheduler = sp_control.MaskerScheduler(conf)

    # init the masker.
    masker = maskers.Masker(
        masker_scheduler=masker_scheduler,
        logger=logger,
        mask_biases=conf.mask_biases,
        structured_masking_info={
            "structured_masking": conf.structured_masking,
            "structured_masking_types": conf.structured_masking_types,
            "force_masking": conf.force_masking,
        },
        threshold=conf.threshold,
        init_scale=conf.init_scale,
        which_ptl=conf.model_type,
        controlled_init=conf.controlled_init,
    )

    # assuming mask all stuff in one transformer block, absorb bert.pooler directly
    weight_types = ["K", "Q", "V", "AO", "I", "O", "P", "E"]

    # parse the get the names of layers to be masked.
    assert conf.layers_to_mask is not None, "Please specify which BERT layers to mask."
    conf.layers_to_mask_ = (
        [int(x) for x in conf.layers_to_mask.split(",")]
        if "," in conf.layers_to_mask
        else [int(conf.layers_to_mask)]
    )
    names_tobe_masked = set()
    names_tobe_masked = maskers.chain_module_names(
        conf.model_type, conf.layers_to_mask_, weight_types
    )
    if conf.mask_classifier:
        if conf.type == "bert" or conf.model_type == "distilbert":
            names_tobe_masked.add("classifier")
        elif conf.model_type == "roberta":
            if (
                conf.model_scheme == "postagging"
                or conf.model_scheme == "multiplechoice"
            ):
                names_tobe_masked.add("classifier")
            elif conf.model_scheme == "vector_cls_sentence":
                names_tobe_masked.add("classifier.dense")
                names_tobe_masked.add("classifier.out_proj")

    # patch modules.
    masker.patch_modules(
        model=model,
        names_tobe_masked=names_tobe_masked,
        name_of_masker=conf.name_of_masker,
    )
    return masker


def do_eval(student_model, teacher_model, eval_dataloader, device, n_gpu, head_mask, ffn_mask, args, loss_fn):
    eval_loss = 0
    nb_eval_steps = 0
    result = {}
    loss_mse = MSELoss()
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        rep_loss, rela_loss, loss = 0., 0., 0.
        with torch.no_grad():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch_
            if args.model_type=='roberta':
                segment_ids = None
            student_atts, student_reps = student_model(input_ids, segment_ids, input_mask, head_mask=head_mask, ffn_mask=ffn_mask)
            teacher_reps, teacher_atts, _ = teacher_model(input_ids, segment_ids, input_mask)

        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]  # speedup 1.5x
        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)

        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
        new_student_reps = student_reps

        if args.repr_distill:
            if 'last_layer' in args.rep_loss_type:
                new_student_reps, new_teacher_reps = new_student_reps[-1:], new_teacher_reps[-1:]
            else:
                new_student_reps, new_teacher_reps = new_student_reps[1:], new_teacher_reps[1:]
            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                if ('full' in args.rep_loss_type) or ('last_layer' in args.rep_loss_type):
                    rep_loss += loss_fn(student_rep, teacher_rep)
                elif 'cls' in args.rep_loss_type:
                    rep_loss += loss_fn(student_rep[:, 0, :], teacher_rep[:, 0, :])

        if args.rela_distill:
            if 'last_layer' in args.rela_loss_type:
                new_student_reps, new_teacher_reps = new_student_reps[-1:], new_teacher_reps[-1:]
            else:
                new_student_reps, new_teacher_reps = new_student_reps[1:], new_teacher_reps[1:]
            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                if 'token' in args.rela_loss_type:
                    attn_mask = input_mask.unsqueeze(1)
                    student_rela = torch.matmul(student_rep, student_rep.transpose(-1, -2))
                    teacher_rela = torch.matmul(teacher_rep, teacher_rep.transpose(-1, -2))
                    student_rela = (student_rela / math.sqrt(student_rep.size(-1))) * attn_mask
                    teacher_rela = (teacher_rela / math.sqrt(teacher_rep.size(-1))) * attn_mask
                elif 'sample' in args.rela_loss_type:
                    student_rela = torch.matmul(student_rep[:, 0, :], student_rep[:, 0, :].transpose(-1, -2))
                    teacher_rela = torch.matmul(teacher_rep[:, 0, :], teacher_rep[:, 0, :].transpose(-1, -2))
                    student_rela = (student_rela / math.sqrt(student_rep.size(-1)))
                    teacher_rela = (teacher_rela / math.sqrt(teacher_rep.size(-1)))

                if ('full' in args.rela_loss_type) or ('last_layer' in args.rela_loss_type):
                    rela_loss += loss_mse(student_rela, teacher_rela)
                elif 'cls' in args.rela_loss_type:
                    rela_loss += loss_mse(student_rela[:, 0, :], teacher_rela[:, 0, :])

        rep_loss = rep_loss / len(new_student_reps)
        rela_loss = rela_loss / len(new_student_reps)
        loss = rep_loss + rela_loss
        if n_gpu > 1:
            rep_loss = rep_loss.mean()
            rela_loss = rela_loss.mean()
        eval_loss += loss.item()
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    result['eval_loss'] = eval_loss
    return result


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    if len(tokens) > max_seq_length:
        logger.info('len(tokens): {}'.format(len(tokens)))
        logger.info('tokens: {}'.format(tokens))
        tokens = tokens[:max_seq_length]

    if len(tokens) != len(segment_ids):
        #logger.info('tokens: {}\nsegment_ids: {}'.format(tokens, segment_ids))
        segment_ids = [0] * len(tokens)

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    if isinstance(tokenizer, BertTokenizer):    # BERT
        input_array = np.zeros(max_seq_length, dtype=np.int)
    elif isinstance(tokenizer, RobertaTokenizer):    #RoBERTa
        input_array = np.ones(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        if hasattr(tokenizer, 'vocab'):
            self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = int(epoch % num_data_epochs)
        logger.info('training_path: {}'.format(training_path))
        data_file = training_path / "epoch_{}.json".format(self.data_epoch)
        metrics_file = training_path / "epoch_{}_metrics.json".format(self.data_epoch)

        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file))

        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path('/cache')
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)

        logging.info("Loading training examples for epoch {}".format(epoch))

        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next

        # assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(int(self.is_nexts[item])))


def reset_threshold(model, init_sparsity):
    thresholds = []
    for name, module in model.named_modules():
        if hasattr(module, 'threshold'):
            _num_zero_element = int(module.weight.nelement() * init_sparsity)
            module.threshold = torch.kthvalue(input=module.weight_mask.data.view(-1), k=_num_zero_element).values
            thresholds.append(module.threshold)
    return float(torch.tensor(thresholds).mean())


def freeze_params(model):
    for name, param in model.named_parameters():
        #if not 'fit_dense' in name:
        #    param.requires_grad = False
        param.requires_grad = False
    return model

#def get_init_scales(init_sparsity, init_scale, threshold):
#    s = (init_scale + threshold) / init_sparsity - init_scale
#    return (-init_scale, s)


def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_mask_weight(args, device, component_type):
    """
    component_type = 'head' or 'ffn'
    """
    mask_dir = args.load_head_mask_dir if component_type=='head' else args.load_ffn_mask_dir
    logger.info("Loading %s score from %s"%(component_type, mask_dir))
    score = torch.from_numpy(np.load(mask_dir))
    # Rescale
    #score =(score - score.min()) / (score.max() - score.min()) * args.init_scale
    _bool_masks = (
        score
        > torch.kthvalue(input=score.view(-1), k=int(score.numel()*args.zero_rate)).values
    )
    score[_bool_masks] = 2.0 * 1e-2
    score[~_bool_masks] = 0.0 * 1e-2
    mask_weight = torch.nn.Parameter(score.to(device))
    return mask_weight


def keep_orig_mask(args, mask, component_type):
    orig_mask = mask.detach()
    tmp_mask = mask.detach()
    if not os.path.exists(args.output_mask_dir):
        os.makedirs(args.output_mask_dir)
    logger.info("Saving model mask to %s", args.output_mask_dir)
    np.save(args.output_mask_dir+'/orig_%s_mask.npy'%component_type, orig_mask.cpu().numpy())
    return orig_mask, tmp_mask

def log_mask_info(args, mask, orig_mask, tmp_mask, result, component_type):
    logger.info("Saving %s mask to %s"%(component_type, args.output_mask_dir))
    np.save(args.output_mask_dir+'/%s_mask.npy'%component_type, mask.detach().cpu().numpy())
    zero_rate = (mask==0).sum().view(-1).div(float(mask.numel())).item()
    mask_distance = (mask!=orig_mask).view(-1).sum().div(float(mask.numel())).item()
    mask_change = (mask!=tmp_mask).view(-1).sum().div(float(mask.numel())).item()
    #mask_changes = [(m!=tm).sum().div(float(m.numel())).item() for m, tm in zip(mask, tmp_mask)]
    #print(mask_changes)
    result['%s_mask_distance'%component_type], result['%s_mask_change'%component_type], result['%s_zero_rate'%component_type] = mask_distance, mask_change, zero_rate
    tmp_mask = mask.detach()
    return result, tmp_mask


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--pregenerated_data",
                        type=Path,
                        required=True)
    parser.add_argument("--pregenerated_eval_data",
                        type=Path,
                        default=None)
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--output_mask_dir",
                        default=None,
                        type=str)
    parser.add_argument("--load_head_mask_dir",
                        default=None,
                        type=str,
                        help="The directory to load head mask for initialization.")
    parser.add_argument("--load_ffn_mask_dir",
                        default=None,
                        type=str,
                        help="The directory to load FFN mask for initialization.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--lr_schedule",
                        default='warmup_linear',
                        type=str,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay',
                        '--wd',
                        default=1e-4,
                        type=float, metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--mask_seed',
                        type=int,
                        default=1,
                        help="seed for load random mask")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    # Additional arguments.
    parser.add_argument("--zero_rate",
                        default=0.,
                        type=float,
                        help="The percentate of 0 in model weights.")
    parser.add_argument("--root_dir",
                        default=None,
                        type=str,
                        help="The root directory.")
    parser.add_argument("--threshold",
                        default=1e-2,
                        type=float,
                        help="The threshold for masking.")
    parser.add_argument("--init_scale",
                        default=2e-2,
                        type=float,
                        help="For initialization the real-value mask matrices.")
    parser.add_argument("--mask_biases",
                        default=False,
                        type=str2bool,
                        help="Whether to mask biases.")
    parser.add_argument("--mask_classifier",
                        default=False,
                        type=str2bool,
                        help="Whether to mask classifier weights.")
    parser.add_argument("--controlled_init",
                        default=None,
                        type=str,
                        choices=["magnitude", "uniform", "magnitude_and_uniform", "double_uniform"],
                        help="To use magnitude pruning or random pruning. mag or rand")
    parser.add_argument("--model_type",
                        default='bert',
                        type=str,
                        help="The ?")
    parser.add_argument("--rela_loss_type",
                        default='full_token',
                        type=str,
                        choices=["full_token", "full_sample", "last_layer_token", "last_layer_sample"],
                        help="The type of rela loss")
    parser.add_argument("--rep_loss_type",
                        default='full_mse',
                        type=str,
                        choices=["full_mse", "full_cosine", "cls_mse", "cls_cosine", "last_layer_mse", "last_layer_cosine"],
                        help="The type of repr loss")
    parser.add_argument("--force_masking",
                        default='bert',
                        type=str,
                        choices=["all", "bert", "classifier"],
                        help="The ?")
    parser.add_argument("--structured_masking",
                        default=None,
                        type=str,
                        help="Whether to perform structured masking.")
    parser.add_argument("--structured_masking_types",
                        default=None,
                        type=str,
                        help="The type of structured masking.")
    parser.add_argument("--name_of_masker",
                        default='MaskedLinear1',
                        type=str,
                        help="To type of masker to use.")
    parser.add_argument("--layers_to_mask",
                        default='0,1,2,3,4,5,6,7,8,9,10,11',
                        type=str,
                        help="The layers to mask.")
    parser.add_argument("--masking_scheduler_conf",
                        default='lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,sparsity_warmup_interval_epoch=0.1,init_epoch=0,final_epoch=1',
                        type=str,
                        help="Configurations for making scheduler.")
    parser.add_argument('--save_step',
                        type=int,
                        default=0)
    parser.add_argument('--eval_step',
                        type=int,
                        default=1000)
    parser.add_argument('--max_step',
                        type=int,
                        default=0)
    parser.add_argument("--rela_distill",
                        default=False,
                        type=str2bool,
                        help="Whether to perform relation distillation.")
    parser.add_argument("--repr_distill",
                        default=True,
                        type=str2bool,
                        help="Whether to perform representation distillation.")
    parser.add_argument("--attn_distill",
                        default=True,
                        type=str2bool,
                        help="Whether to perform attention distillation.")
    parser.add_argument("--load_struc_mask_type",
                        default=None,
                        type=str,
                        help="The masking type for initialization",
                        choices=["random", "one_step", "iterative"])
    parser.add_argument("--load_mask",
                        default=False,
                        type=str2bool,
                        help="Whether to load mask for initialization.")
    parser.add_argument("--structured",
                        default=False,
                        type=str2bool,
                        help="Whether to use structured pruning.")
    parser.add_argument("--mask_dataset",
                        default=None,
                        type=str,
                        help="The dataset for training mask.")
    parser.add_argument("--overwrite_output_dir",
                        default=False,
                        type=str2bool,
                        help="Whether to overwrite output directory.")
    parser.add_argument("--train_ffn_mask",
                        default=False,
                        type=str2bool,
                        help="Whether to train FFN mask.")
    parser.add_argument("--train_head_mask",
                        default=False,
                        type=str2bool,
                        help="Whether to train attention head mask.")


    # This is used for running on Huawei Cloud.
    parser.add_argument('--data_url',
                        type=str,
                        default="")

    args = parser.parse_args()
    logger.info('args:{}'.format(args))

    samples_per_epoch = []
    for i in range(int(args.num_train_epochs)):
        epoch_file = args.pregenerated_data / "epoch_{}.json".format(i)
        metrics_file = args.pregenerated_data / "epoch_{}_metrics.json".format(i)
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print("Warning! There are fewer epochs of pregenerated data ({}) than training epochs ({}).".format(i, args.num_train_epochs))
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.num_train_epochs

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.logging_dir = os.path.join(args.output_dir, 'logging')
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir)

    args.do_lower_case = True if not '-cased' in args.teacher_model else False

    fw_args = open(args.output_dir + '/args.txt', 'w')
    fw_args.write(str(args)+'\n\n')
    fw_args.close()

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case) if args.model_type=='bert' \
            else RobertaTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(int(args.num_train_epochs)):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


    student_model = TinyBertForPreTraining.from_pretrained(args.student_model) if args.model_type=='bert' \
            else RobertaForPreTraining.from_pretrained(args.student_model)

    if args.structured:
        threshold_fn1 = _Binarizer1().apply
        threshold_fn2 = _Binarizer2().apply
        head_mask_weight, ffn_mask_weight = None, None
        if args.load_mask:
            if args.train_head_mask:
                head_mask_weight = init_mask_weight(args, device, 'head')
            if args.train_ffn_mask:
                ffn_mask_weight = init_mask_weight(args, device, 'ffn')
        else:
            logger.info("Initializing random mask...")
            n_layers, n_heads = student_model.bert.config.num_hidden_layers, student_model.bert.config.num_attention_heads
            init_scales = (-args.init_scale, args.init_scale)
            head_mask_weight = torch.nn.Parameter(torch.empty((n_layers, n_heads)).uniform_(*init_scales).to(device))
        student_model = freeze_params(student_model)
        masker = None
    else:
        masker = init_masker(args, student_model, logger)
        zero_rate = save_model_mask(student_model, args.output_mask_dir+'/epoch0', args.model_type)


    teacher_model = BertModel.from_pretrained(args.teacher_model) if args.model_type=='bert' \
            else RobertaModel.from_pretrained(args.teacher_model)
    student_model.to(device)
    teacher_model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        teacher_model = DDP(teacher_model)
    elif n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    size = 0
    for n, p in student_model.named_parameters():
        #logger.info('n: {}'.format(n))
        #logger.info('p: {}'.format(p.nelement()))
        size += p.nelement()

    logger.info('Total parameters: {}'.format(size))

    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.structured:
        mask_weights = [head_mask_weight, ffn_mask_weight]
        optimizer_grouped_parameters = [
                {'params': [w for w in mask_weights if w is not None], 'weight_decay': 0.0}
                ]
        #optimizer_grouped_parameters = [
        #        {'params': [w for w in mask_weights if w is not None], 'weight_decay': 0.0},
        #        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and ('fit_dense' in n)], 'weight_decay': 0.01},
        #        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay)) and ('fit_dense' in n)], 'weight_decay': 0.0}
        #        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         schedule=args.lr_schedule,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    loss_mse = MSELoss()
    if 'mse' in args.rep_loss_type:
        loss_fn = MSELoss()
    elif 'cosine' in args.rep_loss_type:
        loss_fn = CosineLoss()
    tb_writer = SummaryWriter(log_dir=args.logging_dir) if is_tensorboard_available() else None
    if args.do_eval:
        eval_dataset = PregeneratedDataset(epoch=0, training_path=args.pregenerated_eval_data, tokenizer=tokenizer,
                                                num_data_epochs=1, reduce_memory=args.reduce_memory)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info("  Num examples = {}".format(total_train_examples))
    logging.info("  Batch size = %d", args.train_batch_size)
    tr_loss = 0.
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_rela_loss = 0.
    tr_cls_rep_loss = 0.
    nb_tr_examples, nb_tr_steps = 0, 0
    best_tr_loss = float("Inf")

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            #train_sampler = SequentialSampler(epoch_dataset)
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        #student_model.eval()
        student_model.train()
        #teacher_model.eval()
        with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch)) as pbar:
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                if args.model_type=='roberta':
                    segment_ids = None
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                rela_loss = 0.
                cls_rep_loss = 0.

                head_mask, ffn_mask = None, None
                if args.structured:
                    if args.train_head_mask:
                        head_mask = threshold_fn2(head_mask_weight, int(head_mask_weight.numel()*args.zero_rate), head_mask_weight.data.size(1))
                        if global_step == 0:
                            orig_head_mask, tmp_head_mask = keep_orig_mask(args, head_mask, component_type='head')
                    if args.train_ffn_mask:
                        ffn_threshold = torch.kthvalue(input=ffn_mask_weight.view(-1), k=int(ffn_mask_weight.numel()*args.zero_rate)).values.detach()
                        ffn_mask = threshold_fn1(ffn_mask_weight, ffn_threshold)
                        #ffn_mask = threshold_fn2(ffn_mask_weight, int(ffn_mask_weight.numel()*args.zero_rate), ffn_mask_weight.data.size(1))
                        if global_step == 0:
                            orig_ffn_mask, tmp_ffn_mask = keep_orig_mask(args, ffn_mask, component_type='ffn')

                student_atts, student_reps = student_model(input_ids, segment_ids, input_mask, head_mask=head_mask, ffn_mask=ffn_mask)
                with torch.no_grad():
                    teacher_reps, teacher_atts, _ = teacher_model(input_ids, segment_ids, input_mask)
                teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]  # speedup 1.5x
                teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]

                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)

                if args.attn_distill:
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                  teacher_att)
                        att_loss += loss_mse(student_att, teacher_att)

                new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                new_student_reps = student_reps

                if args.repr_distill:
                    if 'last_layer' in args.rep_loss_type:
                        new_student_reps, new_teacher_reps = new_student_reps[-1:], new_teacher_reps[-1:]
                    else:
                        new_student_reps, new_teacher_reps = new_student_reps[1:], new_teacher_reps[1:]

                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        if ('full' in args.rep_loss_type) or ('last_layer' in args.rep_loss_type):
                            rep_loss += loss_fn(student_rep, teacher_rep)
                        elif 'cls' in args.rep_loss_type:
                            rep_loss += loss_fn(student_rep[:, 0, :], teacher_rep[:, 0, :])

                if args.rela_distill:
                    if 'last_layer' in args.rela_loss_type:
                        new_student_reps, new_teacher_reps = new_student_reps[-1:], new_teacher_reps[-1:]
                    else:
                        new_student_reps, new_teacher_reps = new_student_reps[1:], new_teacher_reps[1:]

                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        if 'token' in args.rela_loss_type:
                            attn_mask = input_mask.unsqueeze(1)
                            student_rela = torch.matmul(student_rep, student_rep.transpose(-1, -2))
                            teacher_rela = torch.matmul(teacher_rep, teacher_rep.transpose(-1, -2))
                            student_rela = (student_rela / math.sqrt(student_rep.size(-1))) * attn_mask
                            teacher_rela = (teacher_rela / math.sqrt(teacher_rep.size(-1))) * attn_mask
                        elif 'sample' in args.rela_loss_type:
                            student_rela = torch.matmul(student_rep[:, 0, :], student_rep[:, 0, :].transpose(-1, -2))
                            teacher_rela = torch.matmul(teacher_rep[:, 0, :], teacher_rep[:, 0, :].transpose(-1, -2))
                            student_rela = (student_rela / math.sqrt(student_rep.size(-1)))
                            teacher_rela = (teacher_rela / math.sqrt(teacher_rep.size(-1)))

                        #if global_step%args.eval_step==0:
                        #    print('######################')
                        #    print(student_rela.norm(p=2), teacher_rela.norm(p=2))
                        #    print('######################')

                        if ('full' in args.rela_loss_type) or ('last_layer' in args.rela_loss_type):
                            rela_loss += loss_mse(student_rela, teacher_rela)
                        elif 'cls' in args.rela_loss_type:
                            rela_loss += loss_mse(student_rela[:, 0, :], teacher_rela[:, 0, :])


                rep_loss = rep_loss / len(new_student_reps)
                rela_loss = rela_loss / len(new_student_reps)
                cls_rep_loss = cls_rep_loss / len(new_student_reps)
                att_loss = att_loss / len(student_atts)
                loss = att_loss + rep_loss + rela_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    rep_loss = rep_loss / args.gradient_accumulation_steps
                    rela_loss = rela_loss / args.gradient_accumulation_steps
                    att_loss = att_loss / args.gradient_accumulation_steps
                    cls_rep_loss = cls_rep_loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                if args.attn_distill:
                    tr_att_loss += att_loss.item()
                if args.repr_distill:
                    tr_rep_loss += rep_loss.item()
                if args.rela_distill:
                    tr_rela_loss += rela_loss.item()
                #tr_cls_rep_loss += cls_rep_loss.item()

                if args.train_head_mask:
                    head_mask_grad_norm = head_mask_weight.grad.abs().sum().item()
                if args.train_ffn_mask:
                    ffn_mask_grad_norm = ffn_mask_weight.grad.abs().sum().item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)


                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if ((global_step + 1) % args.eval_step == 0) or (global_step==1):
                        mean_loss = tr_loss * args.gradient_accumulation_steps / args.eval_step
                        mean_att_loss = tr_att_loss * args.gradient_accumulation_steps / args.eval_step
                        mean_rep_loss = tr_rep_loss * args.gradient_accumulation_steps / args.eval_step
                        mean_rela_loss = tr_rela_loss * args.gradient_accumulation_steps / args.eval_step
                        tr_loss = 0.
                        tr_att_loss = 0.
                        tr_rep_loss = 0.
                        tr_rela_loss = 0.

                        logger.info("***** Running evaluation *****")
                        logger.info("  Epoch = {} iter {} step".format(epoch, global_step))
                        logger.info("  Batch size = %d", args.eval_batch_size)
                        if args.do_eval:
                            logger.info("  Num examples = %d", len(eval_dataset))
                        lr = optimizer.param_groups[0]['lr'] * \
                        optimizer.param_groups[0]['schedule'].get_lr(global_step)
                        logger.info("  Learning rate = %.2e", lr)

                        student_model.eval()

                        result = {}
                        if args.do_eval:
                            result = do_eval(student_model, teacher_model, eval_dataloader, device, n_gpu,
                                    head_mask, ffn_mask, args, loss_fn)

                        result['global_step'] = global_step
                        result['loss'] = mean_loss
                        result['att_loss'] = mean_att_loss
                        result['rep_loss'] = mean_rep_loss
                        result['rela_loss'] = mean_rela_loss
                        if args.train_head_mask:
                            result['head_mask_grad_norm'] = head_mask_grad_norm
                        if args.train_ffn_mask:
                            result['ffn_mask_grad_norm'] = ffn_mask_grad_norm
                        logging.info("** ** * Saving mask at %d step ** ** * "%global_step)
                        if args.structured:
                            if args.train_head_mask:
                                result, tmp_head_mask = log_mask_info(args, head_mask, orig_head_mask, tmp_head_mask, result, 'head')
                            if args.train_ffn_mask:
                                result, tmp_ffn_mask = log_mask_info(args, ffn_mask, orig_ffn_mask, tmp_ffn_mask, result, 'ffn')
                        else:
                            zero_rate = save_model_mask(student_model, args.output_mask_dir, args.model_type, is_save=False)
                            result['zero_rate'] = zero_rate
                            mean_thre = reset_threshold(student_model, masker.masker_scheduler.init_sparsity)

                        lr = optimizer.param_groups[0]['lr'] * optimizer.param_groups[0]['schedule'].get_lr(global_step)
                        result['lr'] = lr
                        output_eval_file = os.path.join(args.output_dir, "log.txt")
                        with open(output_eval_file, "a") as writer:
                            logger.info("***** Eval results *****")
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))
                            writer.write('\n')
                        if tb_writer is not None:
                            for k, v in result.items():
                                tb_writer.add_scalar(k, v, global_step)
                        student_model.train()

                    if (args.save_step > 0) and ((global_step + 1) % args.save_step == 0):
                        logging.info("** ** * Saving mask at %d step ** ** * "%global_step)
                        tmp_output_dir = args.output_mask_dir+'/step_%d'%global_step
                        if not os.path.exists(tmp_output_dir):
                            os.makedirs(tmp_output_dir)
                        logger.info("Saving model mask to %s", tmp_output_dir)
                        if args.structured:
                            if args.train_head_mask:
                                np.save(tmp_output_dir+'/head_mask.npy', head_mask.detach().cpu().numpy())
                            if args.train_ffn_mask:
                                np.save(tmp_output_dir+'/ffn_mask.npy', ffn_mask.detach().cpu().numpy())
                        else:
                            mean_thre = reset_threshold(student_model, masker.masker_scheduler.init_sparsity)
                            zero_rate = save_model_mask(student_model, tmp_output_dir, args.model_type)

                if args.max_step > 0 and global_step > args.max_step:
                    break
            if args.max_step > 0 and global_step > args.max_step:
                break


            # Save a trained model
            logging.info("** ** * Saving mask at %d epoch ** ** * "%epoch)
            if args.structured:
                epoch_output_dir  = args.output_mask_dir+'/epoch%d'%epoch
                if not os.path.exists(epoch_output_dir):
                    os.makedirs(epoch_output_dir)
                logger.info("Saving model mask to %s", epoch_output_dir)
                if args.train_head_mask:
                    np.save(epoch_output_dir+'/head_mask.npy', head_mask.detach().cpu().numpy())
                if args.train_ffn_mask:
                    np.save(epoch_output_dir+'/ffn_mask.npy', ffn_mask.detach().cpu().numpy())
            else:
                reset_threshold(student_model, masker.masker_scheduler.init_sparsity)
                #save_model_mask(student_model, args.output_mask_dir+'/epoch%d'%(epoch+1))
            if tb_writer is not None:
                tb_writer.close()

if __name__ == "__main__":
    main()
