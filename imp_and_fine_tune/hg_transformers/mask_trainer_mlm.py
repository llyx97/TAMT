import json
import logging
import math
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from .data.data_collator import DataCollator, DefaultDataCollator
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from .training_args import TrainingArguments, is_tpu_available
from masking.maskers import Masker


try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None
    masker: Masker = None
    head_mask_weight = None
    ffn_mask_weight = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        model_args,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
        masker: Masker = None,
        head_mask_weight = None,
        ffn_mask_weight = None,
        threshold_fn_head = None,
        threshold_fn_ffn = None
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.masker = masker
        self.head_mask_weight = head_mask_weight
        self.ffn_mask_weight = ffn_mask_weight
        self.args = args
        self.model_args = model_args
        self.threshold_fn_head = threshold_fn_head
        self.threshold_fn_ffn = threshold_fn_ffn
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        if self.model_args.structured:
            mask_weights = [self.head_mask_weight, self.ffn_mask_weight]
            optimizer_grouped_parameters = [
                {'params': [w for w in mask_weights if w is not None], 'weight_decay': 0.0},
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and ('predictions' in n or 'classifier' in n)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and ('predictions' in n or 'classifier' in n)],
                    "weight_decay": 0.0,
                },
                    ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        self.optimizers = optimizer, scheduler
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
        # keep track of model topology and gradients
        if os.getenv("WANDB_WATCH") != "false":
            wandb.watch(
                self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def keep_orig_mask(self, mask, component_type):
        orig_mask = mask.detach()
        tmp_mask = mask.detach()
        if not os.path.exists(self.model_args.output_mask_dir):
            os.makedirs(self.model_args.output_mask_dir)
        logger.info("Saving model mask to %s", self.model_args.output_mask_dir)
        np.save(self.model_args.output_mask_dir+'/orig_%s_mask.npy'%component_type, orig_mask.cpu().numpy())
        return orig_mask, tmp_mask

    def log_mask_info(self, mask, orig_mask, tmp_mask, result, component_type):
        logger.info("Saving %s mask to %s"%(component_type, self.model_args.output_mask_dir))
        np.save(self.model_args.output_mask_dir+'/%s_mask.npy'%component_type, mask.detach().cpu().numpy())
        zero_rate = (mask==0).sum().view(-1).div(float(mask.numel())).item()
        mask_distance = (mask!=orig_mask).view(-1).sum().div(float(mask.numel())).item()
        mask_change = (mask!=tmp_mask).view(-1).sum().div(float(mask.numel())).item()
        result['%s_mask_distance'%component_type], result['%s_mask_change'%component_type], result['%s_zero_rate'%component_type] = mask_distance, mask_change, zero_rate
        tmp_mask = mask.detach()
        return result, tmp_mask

    def reset_threshold(self, model, init_sparsity):
        thresholds = []
        for name, module in model.named_modules():
            if hasattr(module, 'threshold'):
                _num_zero_element = int(module.weight.nelement() * init_sparsity)
                module.threshold = torch.kthvalue(input=module.weight_mask.data.view(-1), k=_num_zero_element).values
                thresholds.append(module.threshold)
        return float(torch.tensor(thresholds).mean())

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        #if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            #try:
            #    self.global_step = int(model_path.split("-")[-1].split("/")[0])
            #    epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
            #    steps_trained_in_current_epoch = self.global_step % (
            #        len(train_dataloader) // self.args.gradient_accumulation_steps
            #    )

            #    logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            #    logger.info("  Continuing training from epoch %d", epochs_trained)
            #    logger.info("  Continuing training from global step %d", self.global_step)
            #    logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            #except ValueError:
            #    self.global_step = 0
            #    logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        best_eval_loss = 100.
        best_score = 0.
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if self.model_args.structured:
                    if self.model_args.train_head_mask:
                        head_mask = self.threshold_fn_head(self.head_mask_weight, int(self.head_mask_weight.numel()*self.model_args.zero_rate), self.head_mask_weight.data.size(1))
                        inputs['head_mask'] = head_mask
                        if self.global_step==0:
                            orig_head_mask, tmp_head_mask = self.keep_orig_mask(head_mask, component_type='head')
                    else:
                        head_mask = None
                    if self.model_args.train_ffn_mask:
                        ffn_threshold = torch.kthvalue(input=self.ffn_mask_weight.view(-1), k=int(self.ffn_mask_weight.numel()*self.model_args.zero_rate)).values.detach()
                        ffn_mask = self.threshold_fn_ffn(self.ffn_mask_weight, ffn_threshold)
                        inputs['ffn_mask'] = ffn_mask
                        if self.global_step==0:
                            orig_ffn_mask, tmp_ffn_mask = self.keep_orig_mask(ffn_mask, component_type='ffn')
                    else:
                        ffn_mask = None


                tr_loss += self._training_step(model, inputs, optimizer)


                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if not hasattr(self.optimizers[0], 'accumulate_grad'):
                        if is_tpu_available():
                            xm.optimizer_step(optimizer)
                        else:
                            optimizer.step()
                        scheduler.step()

                    #model.zero_grad()
                    optimizer.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)



                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss
                        if self.model_args.structured:
                            if self.model_args.train_head_mask:
                                logs, tmp_head_mask = self.log_mask_info(head_mask, orig_head_mask, tmp_head_mask, logs, 'head')
                            if self.model_args.train_ffn_mask:
                                logs, tmp_ffn_mask = self.log_mask_info(ffn_mask, orig_ffn_mask, tmp_ffn_mask, logs, 'ffn')
                        elif self.masker is not None:
                            mean_thresh = self.reset_threshold(model, self.masker.masker_scheduler.init_sparsity)

                        #if hasattr(model, 'cls'):
                        #    if model.cls.predictions.bias.requires_grad:
                        #        torch.save(model.cls.predictions, self.model_args.output_mask_dir+'/mlm_head.bin')
                        #if hasattr(model, 'classifier'):
                        #    if model.classifier.bias.requires_grad:
                        #        torch.save(model.classifier, self.model_args.output_mask_dir+'/classifier.bin')

                        self._log(logs)

                        if self.args.evaluate_during_training:
                            results = self.evaluate()
                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                if key=='eval_acc' or key=='eval_mcc' or key=='eval_pearson':
                                    if hasattr(self.model, 'classifiers') and int(key.split('_')[-1]) < len(self.model.classifiers)-1:
                                        continue
                                    if best_score<value:
                                        best_score = value
                                        results_at_best_score = results
                                        self.save_struc_model_mask(self.args.output_dir+'/best_eval_mask-%d'%self.global_step, head_mask, ffn_mask)
                                        self.save_struc_model_mask(self.args.output_dir+'/best_eval_mask', head_mask, ffn_mask)
                                        if hasattr(model, 'classifier'):
                                            torch.save(model.classifier, self.args.output_dir+'/best_eval_mask/classifier.bin')
                                    elif (value==0) and (best_score==0):
                                        results_at_best_score = results
                                        self.save_struc_model_mask(self.args.output_dir+'/best_eval_mask', head_mask, ffn_mask)
                                    self._log({'best_score': best_score})
                                elif (self.masker is not None) and 'eval_loss' in eval_key:
                                    if hasattr(self.model, 'classifiers') and int(key.split('_')[-1]) < len(self.model.classifiers)-1:
                                        continue
                                    if best_eval_loss>value:
                                        best_eval_loss = value
                                        results_at_best_score = results
                                        mean_thresh = self.reset_threshold(model, self.masker.masker_scheduler.init_sparsity)
                                        zero_rate = self.save_model_mask(self.args.output_dir+'/best_eval_mask')
                                        if hasattr(model, 'cls'):
                                            torch.save(model.cls.predictions, self.args.output_dir+'/best_eval_mask/mlm_head.bin')
                                    self._log({'best_score': best_eval_loss})


                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        #self.save_model(output_dir)
                        if self.model_args.structured:
                            self.save_struc_model_mask(output_dir, head_mask, ffn_mask)
                        else:
                            mean_thresh = self.reset_threshold(model, self.masker.masker_scheduler.init_sparsity)
                            zero_rate = self.save_model_mask(output_dir)
                            self._log({'zero_rate': zero_rate})

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        #if is_tpu_available():
                        #    xm.rendezvous("saving_optimizer_states")
                        #    xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        #    xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        #elif self.is_world_master():
                        #    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        #    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step), best_eval_loss, best_score, results_at_best_score

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        if is_wandb_available():
            wandb.log(logs, step=self.global_step)
        for key, value in logs.items():
            logs[key] = np.float(value)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if hasattr(self.optimizers[0], 'accumulate_grad'):
            self.optimizers[0].accumulate_grad()

        return loss.item()

    def is_local_master(self) -> bool:
        if is_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """

        if is_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def save_struc_model_mask(self, output_dir, head_mask=None, ffn_mask=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if head_mask is not None:
            logger.info("Saving model checkpoint to %s/head_mask.npy", output_dir)
            np.save(output_dir+'/head_mask.npy', head_mask.detach().cpu().numpy())
        if ffn_mask is not None:
            logger.info("Saving model checkpoint to %s/ffn_mask.npy", output_dir)
            np.save(output_dir+'/ffn_mask.npy', ffn_mask.detach().cpu().numpy())

    def save_model_mask(self, output_dir: Optional[str] = None):
        mask_dict = {}
        zero_sum, elem_sum = 0., 0.
        logger.info('Collecting mask...')
        for name, module in self.model.named_modules():
            if hasattr(module, 'threshold'):
                mask = module.weight_mask
                mask = self.binarizer_fn1(mask, module.threshold).bool().cpu()
                zero_sum += (mask==0).sum()
                elem_sum += mask.numel()
                mask_dict[name+'.weight'] = mask
        #for key in self.model.state_dict().keys():
        #    if 'mask' in key:
        #        mask = self.model.state_dict()[key]
        #        mask = self.binarizer_fn1(mask, self.masker.threshold).bool().cpu()
        #        zero_sum += (mask==0).sum()
        #        elem_sum += mask.numel()
        #        mask_dict[key.replace('_mask', '')] = mask

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        zero_rate = 100*zero_sum/elem_sum
        logger.info("Saving model mask to %s", output_dir)
        logger.info("Zero rate = %.2f", zero_rate)
        torch.save(mask_dict, os.path.join(output_dir, 'mask.pt'))
        return zero_rate

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        #torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def binarizer_fn1(self, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0.0
        outputs[inputs.gt(threshold)] = 1.0
        return outputs

    def binarizer_fn2(self, inputs):
        outputs = inputs.clone()
        inputs.data.clamp_(-1, 1)
        outputs.data = (torch.sign(outputs.data) + 1) / 2
        return outputs

    def binarizer_fn3(self, inputs):
        # outputs = inputs.clone()
        # outputs = torch.bernoulli(torch.sigmoid(outputs))
        outputs = torch.bernoulli(torch.sigmoid(inputs))
        return outputs

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            if self.model_args.structured:
                if self.model_args.train_head_mask:
                    with torch.no_grad():
                        head_mask = self.threshold_fn_head(self.head_mask_weight, int(self.head_mask_weight.numel()*self.model_args.zero_rate), self.head_mask_weight.data.size(1))
                    inputs['head_mask'] = head_mask
                if self.model_args.train_ffn_mask:
                    ffn_threshold = torch.kthvalue(input=self.ffn_mask_weight.view(-1), k=int(self.ffn_mask_weight.numel()*self.model_args.zero_rate)).values.detach()
                    with torch.no_grad():
                        ffn_mask = self.threshold_fn_ffn(self.ffn_mask_weight, ffn_threshold)
                    inputs['ffn_mask'] = ffn_mask

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output
