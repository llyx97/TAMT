# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
import math
import os 
from dataclasses import dataclass, field
from typing import Optional
import torch.nn.utils.prune as prune
import torch
import numpy as np

from hg_transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def see_weight_rate(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.query.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.query.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.key.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.key.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.self.value.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.self.value.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].attention.output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].attention.output.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].intermediate.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].intermediate.dense.weight == 0))

        sum_list = sum_list+float(model.bert.encoder.layer[ii].output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.bert.encoder.layer[ii].output.dense.weight == 0))


    sum_list = sum_list+float(model.bert.pooler.dense.weight.nelement())
    sum_list = sum_list+float(model.bert.embeddings.word_embeddings.weight.nelement())
    zero_sum = zero_sum+float(torch.sum(model.bert.pooler.dense.weight == 0))
    zero_sum = zero_sum+float(torch.sum(model.bert.embeddings.word_embeddings.weight == 0))


    return 100*zero_sum/sum_list


def pruning_model_with_mask(model, mask_dict, do_remove=False):
    parameters_to_prune =[]
    mask_list = []
    suffix = '.weight_mask' if '_mask' in list(mask_dict.keys())[0] else '.weight'
    for ii in range(12):
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.query)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.query%s'%suffix])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.key)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.key%s'%suffix])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.self.value)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.self.value%s'%suffix])
        parameters_to_prune.append(model.bert.encoder.layer[ii].attention.output.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.attention.output.dense%s'%suffix])
        parameters_to_prune.append(model.bert.encoder.layer[ii].intermediate.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.intermediate.dense%s'%suffix])
        parameters_to_prune.append(model.bert.encoder.layer[ii].output.dense)
        mask_list.append(mask_dict['bert.encoder.layer.'+str(ii)+'.output.dense%s'%suffix])

    parameters_to_prune.append(model.bert.pooler.dense)
    parameters_to_prune.append(model.bert.embeddings.word_embeddings)
    mask_list.append(mask_dict['bert.pooler.dense%s'%suffix])
    mask_list.append(mask_dict['bert.embeddings.word_embeddings%s'%suffix])

    for ii in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list[ii])

    modules_to_prune = [key.replace('.weight', '') for key in mask_dict.keys()]
    if do_remove:
        for name, module in model.named_modules():
            if name in modules_to_prune:
                prune.remove(module, 'weight')


def prune_with_struc_mask(model, mask_dir, component_type):
    """
    component_type = 'head' or 'ffn'
    """
    mask_file = os.path.join(mask_dir, 'ffn_mask.npy') if component_type=='ffn' else os.path.join(mask_dir, 'head_mask.npy')
    logger.info("Loading mask from %s"%mask_file)
    mask = torch.from_numpy(np.load(mask_file))
    to_prune = {}
    for layer in range(len(mask)):
        to_mask = [h[0] for h in (1 - mask[layer].long()).nonzero().tolist()]
        to_prune[layer] = to_mask
    assert sum(len(h) for h in to_prune.values()) == (1 - mask.long()).sum().item()

    logger.info("%s zero rate:%.3f"%(component_type, (mask==0).view(-1).sum().div(float(mask.numel()))))
    if component_type=='head':
        logger.info(f"Pruning heads {to_prune}")
        model.prune_heads(to_prune)
    elif component_type=='ffn':
        model.prune_ffns(to_prune)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    zero_rate: Optional[float] = field(
        default=0., metadata={"help": "The percentate of 0 in model weights."}
    )
    prun_type: Optional[str] = field(
        default='mag', metadata={"help": "To use magnitude pruning or random pruning. mag or rand"}
    )
    mask_seed: Optional[int] = field(
        default=0, metadata={"help": "The seed for random masking."}
    )
    root_dir: Optional[str] = field(
        default=None, metadata={"help": "The root directory."}
    )
    mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to load mask."}
    )
    ffn_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The ffn mask directory."}
    )
    head_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The head mask directory."}
    )
    structured: str2bool = field(
        default=True, metadata={"help": "Whether to use structured pruning."}
    )
    load_mlm_head: str2bool = field(
        default=False, metadata={"help": "Whether to load the mlm head obtained from mask training."}
    )
    prune_ffn: str2bool = field(
        default=False, metadata={"help": "Whether to prune FFNs."}
    )
    prune_head: str2bool = field(
        default=False, metadata={"help": "Whether to prune attention heads."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if not 'wiki' in file_path:
        args.line_by_line = True
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    if model_args.zero_rate > 0:
        if model_args.structured:
            if model_args.prune_head:
                prune_with_struc_mask(model, model_args.head_mask_dir, 'head')
                mlm_head_dir = model_args.head_mask_dir.replace('head_mask.npy', '')
            if model_args.prune_ffn:
                prune_with_struc_mask(model, model_args.ffn_mask_dir, 'ffn')
                mlm_head_dir = model_args.ffn_mask_dir.replace('ffn_mask.npy', '')
            param_count = 0
            for n, p in model.named_parameters():
                param_count += p.nelement()
            param_count /= 1e6
            logger.info("Parameter count:%.2fM"%param_count)
        else:
            mask_seed = model_args.mask_seed if 'rand' in model_args.prun_type else ''
            if model_args.mask_dir is not None:
                mask_dir = model_args.mask_dir
            else:
                mask_dir = os.path.join(model_args.root_dir, 'models/prun_bert',
                        model_args.prun_type, str(model_args.zero_rate), str(mask_seed), 'mask.pt')
            mlm_head_dir =  model_args.mask_dir.replace('mask.pt', '')
            logger.info('Loading mask from %s...'%mask_dir)
            mask = torch.load(mask_dir)
            #print('Mask for bert.encoder.layer.1.attention.self.query.weight: \n')
            #print(mask['bert.encoder.layer.1.attention.self.query.weight'])
            pruning_model_with_mask(model, mask)
            zero = see_weight_rate(model)
            print('model 0:',zero)
        if model_args.load_mlm_head:
            mlm_head = torch.load(os.path.join(mlm_head_dir, 'mlm_head.bin')).to(training_args.device)
            model.cls.predictions = mlm_head

    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer, plm_probability=data_args.plm_probability, max_span_length=data_args.max_span_length,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity, "eval_loss": eval_output["eval_loss"]}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
