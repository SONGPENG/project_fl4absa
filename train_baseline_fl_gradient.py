from fl import BaselineModel
from pytorch_transformers import BertTokenizer

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np
from tqdm import tqdm, trange
import torch

from data_utils import DataProcessor,seq_reduce_ret_len
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

class TerminalInstructor():
    def __init__(self, args, dataset, device):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_processor = DataProcessor(args.data_dir, self.tokenizer,
                                            max_seq_len=args.max_seq_len,
                                            batch_size=args.batch_size)
        self.train_dataloader, self.test_dataloader, self.dev_dataloader = self.data_processor.get_dataloader(dataset)

        if args.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=args.device.index)))

    def fetch_train_batchdata(self):
        while True:
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                yield sample_batched

    def train(self, model, step=1, grad_div=1):
        n_correct, n_total, loss_total = 0, 0, 0
        model.train()
        for _ in range(step):
            sample_batched = next(self.fetch_train_batchdata())
            reduce_seq_len = seq_reduce_ret_len(sample_batched["input_ids"])
            input_ids = sample_batched["input_ids"][:,:reduce_seq_len].to(self.device)
            segment_ids = sample_batched["segment_ids"][:,:reduce_seq_len].to(self.device)
            attention_mask = sample_batched["input_mask"][:,:reduce_seq_len].to(self.device)
            label_ids = sample_batched["label_ids"].to(self.device)

            tag_seq, loss = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, labels=label_ids)
            loss = loss / step / grad_div
            loss.backward()

            n_correct += (tag_seq == label_ids).sum().item()
            n_total += len(tag_seq)
            loss_total += loss.item() * len(tag_seq)
        return n_correct, n_total, loss_total

    def eval(self, model):
        model.eval()
        n_correct, n_total, loss_total, nb_tr_steps = 0, 0, 0, 0
        t_targets_all, t_outputs_all = None, None

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.test_dataloader):
                reduce_seq_len = seq_reduce_ret_len(sample_batched["input_ids"])
                input_ids = sample_batched["input_ids"][:,:reduce_seq_len].to(self.device)
                segment_ids = sample_batched["segment_ids"][:,:reduce_seq_len].to(self.device)
                attention_mask = sample_batched["input_mask"][:,:reduce_seq_len].to(self.device)
                label_ids = sample_batched["label_ids"].to(self.device)

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

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1, 2], average='macro')

        logger.info("dataset: {}, nb_tr_examples: {}, nb_tr_steps: {}, acc: {}, f1: {}".format(self.dataset, n_total, nb_tr_steps, acc, f1))

        return {
            "precision": acc,
            "f1": f1
        }

class Instructor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_processor = DataProcessor(args.data_dir, self.tokenizer,
                                            max_seq_len=args.max_seq_len,
                                            batch_size=args.batch_size)

    def saving_model(self, saving_model_path, model, optimizer):
        if not os.path.exists(saving_model_path):
            os.mkdir(saving_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(saving_model_path, CONFIG_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(model_to_save.config.to_json_string())
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

    def run(self):
        self.save_args()
        args = self.args

        num_labels = self.data_processor.get_tag_size()
        model = BaselineModel.from_pretrained(self.args.bert_model, num_labels=num_labels)
        model._reset_params(self.args.initializer)
        model = model.to(self.args.device)

        terminal_dict = {}
        max_sample_num = 0
        dataset_list = self.args.dataset.split(",")
        print("dataset: {}".format(self.args.dataset))
        for dataset in dataset_list:
            terminal_dict[dataset] = TerminalInstructor(args, dataset, self.args.device)
            sample_num = len(terminal_dict[dataset].train_dataloader) * args.batch_size
            if sample_num > max_sample_num:
                max_sample_num = sample_num
        steps_per_epoch = max(int(max_sample_num / args.batch_size), 1)
        args.total_step = steps_per_epoch * args.num_epoch

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=self.args.total_step)

        results = {"bert_model": args.bert_model, "dataset": args.dataset, "warmup":args.warmup_proportion,
                   "batch_size": args.batch_size * args.world_size * args.gradient_accumulation_steps,
                   "learning_rate": args.learning_rate, "seed": args.seed, "sample_step": args.sample_step,
                   "best_f1": 0, "frac": args.frac}
        for epoch in trange(self.args.num_epoch, desc="Epoch"):
            n_correct, n_total, loss_total = 0,0,0
            for _ in range(steps_per_epoch):
                #randomly select client
                # m = max(int(args.frac * len(dataset_list)), 1)
                # cand_dataset_list = np.random.choice(dataset_list, m, replace=False)
                cand_dataset_list = dataset_list
                for dataset in cand_dataset_list:
                    terminal = terminal_dict[dataset]
                    _n_correct, _n_total, _loss_total = terminal.train(model, step=args.sample_step, grad_div=len(cand_dataset_list))
                    n_correct += _n_correct
                    n_total += _n_total
                    loss_total += _loss_total

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            precisions = []
            f1s = []
            for dataset in dataset_list:
                terminal = terminal_dict[dataset]
                eval_result = terminal.eval(model)
                precisions.append(eval_result["precision"])
                f1s.append(eval_result["f1"])
                results["{}_{}_precision".format(dataset, epoch)] = eval_result["precision"]
                results["{}_{}_f1".format(dataset, epoch)] = eval_result["f1"]
            avg_precision = sum(precisions) / len(precisions)
            avg_f1 = sum(f1s) / len(f1s)
            if avg_f1 > results["best_f1"]:
                results["best_epoch"] = epoch
                results["best_f1"] = avg_f1
                results["best_precision"] = avg_precision
                saving_model_path = os.path.join(self.args.outdir, 'epoch-{}'.format(epoch))
                results["best_checkpoint"] = saving_model_path

                if self.args.save and is_main_process():
                    self.saving_model(saving_model_path, model, optimizer)

            output_eval_file = os.path.join(self.args.outdir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for k, v in results.items():
                    writer.write("{}={}\n".format(k, v))

def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
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
    parser.add_argument("--total_step", type=int, default=1000, help="total step")
    parser.add_argument("--sample_step", type=int, default=1, help="sample step")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    args = parser.parse_args()

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
    args.outdir = os.path.join(args.outdir, "{}_bts_{}_lr_{}_warmup_{}_seed_{}_bert_dropout_{}_{}".format(
        args.dataset,
        args.batch_size,
        args.learning_rate,
        args.warmup_proportion,
        args.seed,
        args.bert_dropout,
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

    log_file = '{}/{}-{}.log'.format(args.log, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()


if __name__ == '__main__':
    main()
