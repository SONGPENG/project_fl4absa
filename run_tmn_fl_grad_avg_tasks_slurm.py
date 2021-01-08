#! coding: utf-8

import os
import argparse
import string
import random
from pathlib import Path
import multiprocessing as mp
import time
import argparse


def run_task(args):
    dataset, lr, bert_model, batchsize, warmup, gpuid, bert_dropout, seed, attention, topic_method = args
    home_path = os.getcwd()
    taskname = dataset

    import datetime
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logfile = "./log/{}_{}_{}_{}.log".format(dataset, lr, batchsize, now_time)

    topic_model_path = os.path.join(home_path, "data/topics/weight_matrix.bin")
    w_vocab_path = os.path.join(home_path, "data/topics/vocab.txt")
    data_dir = os.path.join(home_path, "data")
    output_dir = os.path.join(home_path,
                "results/absa_fl_tmn_fl_grad_{attention}_topic_{topic_method}".format(
                attention=attention, topic_method=topic_method))

    dataset_list = "laptop,rest14,twitter"

    cmd = "python train_tmn_fl_gradient.py --seed {seed} --dataset {dataset_list} --data_dir {data_dir} " \
          "--bert_model {bert_model}  --batch_size {batchsize} --bert_dropout {bert_dropout} --outdir {output_dir} " \
          "--num_epoch 30 --warmup_proportion {warmup} --fp16 --learning_rate {lr} " \
          "--topic_method {topic_method} --attention {attention} " \
          "--topic_model_path {topic_model_path} --w_vocab_path {w_vocab_path} ".format(
        dataset_list=dataset_list,
        data_dir=data_dir,
        lr=lr,
        bert_model=bert_model,
        batchsize=batchsize,
        warmup=warmup,
        home_path=home_path,
        bert_dropout=bert_dropout,
        seed=seed,
        output_dir=output_dir,
        topic_method=topic_method,
        attention=attention,
        topic_model_path=topic_model_path,
        w_vocab_path=w_vocab_path
    )
    cmd = "srun --job-name ft_g_tmn_grd --mem=80GB --cpus-per-task=4 -n 1 --gres=gpu:1 {cmd} >{logfile} 2>&1".format(
        taskname=taskname,
        cmd=cmd,
        logfile=logfile,
        gpu=1
    )
    print(cmd)
    os.system(cmd)
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--datasets", type=str, default="laptop,rest14,rest15,rest16,twitter")
    parser.add_argument("--batchsizes", type=str, default="")
    parser.add_argument("--lrs", type=str, default="")
    parser.add_argument("--bert_model", type=str, default="./bert-large-uncased/")
    parser.add_argument('--embedding', default='embedding', type=str)
    parser.add_argument('--encoder', default='bilstm', type=str)
    parser.add_argument('--bert_dropout', default=0.1, type=float)
    parser.add_argument('--seed', default=40, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--warmups', default=0.06, type=float)
    args = parser.parse_args()

    warmups = [args.warmups]

    task_list = []

    batchsizes = [32]
    lrs = [1e-5, 3e-5, 5e-5]
    topic_methods = [0, 1, 2]
    attentions = ["avg", "loss"]
    for lr in lrs:
        for warmup in warmups:
            for batchsize in batchsizes:
                for attention in attentions:
                    for topic_method in topic_methods:
                        task_list.append([args.datasets, lr, args.bert_model, batchsize, warmup, args.gpu,
                                          args.bert_dropout, args.seed, attention, topic_method])

    if args.processes == 0 :
        args.processes = len(task_list)
    pool = mp.Pool(processes=args.processes)
    pool.map(run_task, task_list)

if __name__ == "__main__":
    main()
