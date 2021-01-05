from ntm import NTMModel
from pytorch_transformers import BertTokenizer

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np
import json

import torch

from data_utils import DataProcessor
from sklearn import metrics
from utils import is_main_process, get_rank

from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler, PolyWarmUpScheduler
from apex import amp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

class Instructor:
    def __init__(self, args):
        self.args = args
        self.dataset_list = args.dataset.split(",")
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_processor = DataProcessor(args.data_dir, self.tokenizer,
                                            max_seq_len=args.max_seq_len,
                                            batch_size=args.batch_size)

        if args.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=args.device.index)))

    def saving_model(self, saving_model_path, model, optimizer):
        if not os.path.exists(saving_model_path):
            os.mkdir(saving_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(saving_model_path, CONFIG_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        torch.save(model_to_save.state_dict(), output_model_file)
        config = {'lr':self.args.learning_rate, 'dataset':self.args.dataset, 'save_dir':self.args.outdir}
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(config))
        torch.save({'optimizer': optimizer.state_dict(),
                    'master params': list(amp.master_params(optimizer))},
                   output_optimizer_file)

    def load_model(self, model, optimizer, saving_model_path):
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        #model
        checkpoint_model = torch.load(output_model_file, map_location="cpu")
        model.load_state_dict(checkpoint_model)
        #optimizer
        checkpoint_optimizer = torch.load(output_optimizer_file, map_location="cpu")
        if self.args.fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint_optimizer['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint_optimizer['master params']):
                param.data.copy_(saved_param.data)
        else:
            optimizer.load_state_dict(checkpoint_optimizer["optimizer"])
        return model, optimizer

    def save_args(self):
        output_args_file = os.path.join(self.args.outdir, 'training_args.bin')
        torch.save(self.args, output_args_file)

    def _train(self, model, optimizer, scheduler, train_data_loader):
        path = None

        nb_tr_steps = 0
        tr_loss = 0
        average_loss = 0
        global_step = 0

        min_loss = 1000
        args = self.args
        results = {"bert_model": args.bert_model, "dataset": args.dataset, "warmup":args.warmup_proportion,
                   "batch_size": args.batch_size * args.world_size * args.gradient_accumulation_steps,
                   "learning_rate": args.learning_rate, "seed": args.seed, "num_layers":args.num_layers}
        for dataset in self.dataset_list:
            results["{}_best_loss".format(dataset)] = 0
        for epoch in range(self.args.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                # clear gradient accumulators
                input_ids = sample_batched
                if args.fp16:
                    input_ids = torch.tensor(input_ids, dtype = torch.half)
                loss = model(input_ids.to(self.args.device))

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss.item()
                average_loss += loss
                nb_tr_steps += 1
                if (i_batch + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    if args.fp16:
                        scheduler.step()

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step % self.args.log_step == 0:
                    train_loss = loss / args.batch_size / args.n_gpu
                    if args.fp16:
                        logger.info('global_step: {}, loss: {:.4f}, '
                                    'lr: {:.6f}'.format(global_step, train_loss, scheduler.get_lr()[0]))
                    else:
                        logger.info('global_step: {}, loss: {:.4f}, '
                                    'lr: {:.6f}'.format(global_step, train_loss, optimizer.get_lr()[0]))

            output_eval_file = os.path.join(self.args.outdir, "eval_results.txt")
            results['{}_epoch_{}_loss'.format(dataset, epoch)] = tr_loss/nb_tr_steps
            if tr_loss/nb_tr_steps < min_loss:
                min_loss = tr_loss/nb_tr_steps
                results["{}_best_loss".format(dataset)] = min_loss
                if self.args.save:
                    self.saving_model(self.args.outdir, model, optimizer)
        with open(output_eval_file, "w") as writer:
            for k, v in results.items():
                writer.write("{}={}\n".format(k, v))
        return path

    """
    def _evaluate_acc_f1(self, model, data_loader):
        model.eval()
        n_correct, n_total, loss_total, nb_tr_steps = 0, 0, 0, 0
        t_targets_all, t_outputs_all = None, None

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                input_ids = sample_batched["input_ids"].to(self.args.device)
                segment_ids = sample_batched["segment_ids"].to(self.args.device)
                attention_mask = sample_batched["input_mask"].to(self.args.device)
                label_ids = sample_batched["label_ids"].to(self.args.device)

                tag_seq = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)

                nb_tr_steps += 1
                n_correct += (tag_seq == label_ids).sum().item()
                n_total += len(tag_seq)

                if t_targets_all is None:
                    t_targets_all = label_ids
                    t_outputs_all = tag_seq
                else:
                    t_targets_all = torch.cat((t_targets_all, label_ids), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, tag_seq), dim=0)

        logger.info("nb_tr_examples: {}, nb_tr_steps: {}".format(n_total, nb_tr_steps))

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1, 2], average='macro')

        return {
            "precision": acc,
            "f1": f1
        }
        """
        
    def run(self):
        self.save_args()
        
        train_dataloader = self.data_processor.get_all_train_dataloader(self.dataset_list)

        vocab_size = self.data_processor.get_vocab_size()
        model = NTMModel(vocab_size=vocab_size, topic_size=4)
        #model._reset_params(self.args.initializer)
        model = model.to(self.args.device)

        num_train_optimization_steps = int(
            len(train_dataloader) / self.args.gradient_accumulation_steps) * self.args.num_epoch

        print("trainset: {}, batch_size: {}, gradient_accumulation_steps: {}, num_epoch: {}, num_train_optimization_steps: {}".format(
            len(train_dataloader) * self.args.batch_size, self.args.batch_size, self.args.gradient_accumulation_steps,
            self.args.num_epoch, num_train_optimization_steps))

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if self.args.fp16:
            print("using fp16")
            try:
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=self.args.learning_rate,
                                  bias_correction=False)

            if self.args.loss_scale == 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale=self.args.loss_scale)
            scheduler = LinearWarmUpScheduler(optimizer, warmup=self.args.warmup_proportion,
                                              total_steps=num_train_optimization_steps)
        else:
            print("using fp32")
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=self.args.learning_rate,
                                 warmup=self.args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
            scheduler = None

        if self.args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        self._train(model, optimizer, scheduler, train_dataloader)


def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--test_dataset', default=None, type=str)
    parser.add_argument('--data_dir', default='ATB', type=str)
    parser.add_argument('--embedding', default='embedding', type=str)
    parser.add_argument('--encoder', default='bilstm', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default='2e-5', type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--log', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=1024, type=int)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=50, type=int)
    parser.add_argument('--bert_model', default='./bert-large-uncased', type=str)
    parser.add_argument('--outdir', default='./', type=str)
    parser.add_argument('--tool', default='stanford', type=str)
    parser.add_argument('--warmup_proportion', default=0.06, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--loss_scale', default=0, type=int)
    parser.add_argument('--save', action='store_true', help="Whether to save model")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size", type=int, default=1, help="local_rank for distributed training on gpus")
    parser.add_argument("--init_method", type=str, default="", help="init_method")
    args = parser.parse_args()

    if 'SLURM_NTASKS' in os.environ:
        args.world_size = int(os.environ['SLURM_NTASKS'])
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
    if 'SLURM_LOCALID' in os.environ:
        args.local_rank = int(os.environ['SLURM_LOCALID'])
    if 'SLURM_JOB_ID' in os.environ:
        jobid = os.environ['SLURM_JOB_ID']
        sharefile = os.path.join(args.outdir, "{}.sharefile".format(jobid))
        args.init_method = "file://{}".format(sharefile)

    args.initializer = torch.nn.init.xavier_uniform_

    return args

def main():
    args = get_args()

    import datetime
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(args.outdir):
        try:
            os.mkdir(args.outdir)
        except Exception as e:
            print(str(e))
    args.outdir = os.path.join(args.outdir, "{}_bts_{}_lr_{}_warmup_{}_seed_{}_{}".format(
        args.dataset,
        args.batch_size,
        args.learning_rate,
        args.warmup_proportion,
        args.seed,
        now_time
    ))
    if not os.path.exists(args.outdir):
        try:
            os.mkdir(args.outdir)
        except Exception as e:
            print(str(e))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                             world_size=args.world_size)
    args.device = device
    args.n_gpu = n_gpu
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}, init_method: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16, args.init_method))

    log_file = '{}/{}-{}-{}.log'.format(args.log, args.dataset, "NTM", strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()


if __name__ == '__main__':
    main()
