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
    dataset, lr, bert_model, batchsize, warmup, gpuid, bert_dropout, seed, encoder, multi_criteria, adversary = args
    home_path = os.getcwd()
    taskname = dataset

    import datetime
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logfile = "./log/{}_{}_{}_{}.log".format(dataset, lr, batchsize, now_time)

    data_dir = os.path.join(home_path, "data")
    output_dir = "{home_path}/results/absa_fl_baseline_fl_weight_avg".format(
        home_path=home_path
    )

    dataset_list = "laptop"

    cmd = "python train_NTM.py --seed {seed} --dataset {dataset} --data_dir {data_dir} " \
          "--bert_model {bert_model}  --batch_size {batchsize} --bert_dropout {bert_dropout} --outdir {output_dir} " \
          "--num_epoch 30 --warmup_proportion {warmup} --fp16 --learning_rate {lr} --gpus {gpuid} ".format(
        dataset=dataset,
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
        gpuid=gpuid
    )
    cmd = "{cmd} >{logfile} 2>&1".format(
        taskname=taskname,
        cmd=cmd,
        logfile=logfile,
        gpuid=gpuid
    )
    print(cmd)
    os.system(cmd)
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--datasets", type=str, default="laptop")
    parser.add_argument("--batchsizes", type=str, default="")
    parser.add_argument("--lrs", type=str, default="")
    parser.add_argument("--bert_model", type=str, default="C:\\Users\\qhdnf\\Desktop\\School\\Lab\\NLP\\Federated Learning\\project_ABSA_FL\\bert-large-uncased")
    parser.add_argument('--embedding', default='embedding', type=str)
    parser.add_argument('--encoder', default='bilstm', type=str)
    parser.add_argument('--bert_dropout', default=0.1, type=float)
    parser.add_argument('--seed', default=40, type=int)
    parser.add_argument('--gpu', default="8", type=str)
    parser.add_argument('--multi_criteria', default=True, type=bool)
    parser.add_argument('--adversary', action="store_true", help="whether use adversary")
    parser.add_argument('--warmups', default=0.06, type=float)
    args = parser.parse_args()

    warmups = [args.warmups]

    task_list = []

    batchsizes = [8]
    lrs = [1e-5]
    for lr in lrs:
        for warmup in warmups:
            for batchsize in batchsizes:
                task_list.append([args.datasets, lr, args.bert_model, batchsize, warmup, args.gpu,
                                  args.bert_dropout, args.seed, args.encoder, args.multi_criteria, args.adversary])

    if args.processes == 0 :
        args.processes = len(task_list)
    pool = mp.Pool(processes=args.processes)
    pool.map(run_task, task_list)

if __name__ == "__main__":
    main()
