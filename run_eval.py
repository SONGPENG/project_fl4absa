#! coding: utf-8

import os
import json
import datetime
import argparse
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict
import re

def get_eval_result(task_name, eval_result_path):
    eval_result_file = eval_result_path / "eval_results.txt"
    res = {}
    with open(eval_result_file, 'r') as f:
        for line in f:
            k, v = line.strip().split("=", 1)
            if ("precision" in k or "recall" in k or "f1" in k):
                res[k] = float(v)
            else:
                res[k] = v
    if "best_checkpoint" in res:
        best_epoch = res["best_checkpoint"].split("-")[-1]
    else:
        best_epoch = res["{}_best_checkpoint".format(task_name)].split("-")[-1]
        res["best_checkpoint"] = res["{}_best_checkpoint".format(task_name)]
    if True:
        weight_dict = {
            "laptop":0.25,
            "rest14":0.25,
            "twitter":0.5
        }
        best_epoch = 0
        best_score = 0
        for i in range(30):
            score = sum([float(res["{}_{}_f1".format(t, i)])*w for t,w in weight_dict.items()])
            if score > best_score:
                best_score = score
                best_epoch = i
        res["best_score"] = best_score
        res["best_checkpoint"] = "epoch-{}".format(best_epoch)
    if "{}_{}_test_f1".format(task_name, best_epoch) in res or "{}_{}_test_precision".format(task_name, best_epoch) in res:
        res["test_f1"] = float(res["{}_{}_test_f1".format(task_name, best_epoch)])
        res["test_precision"] = float(res["{}_{}_test_precision".format(task_name, best_epoch)])
    elif "{}_{}_f1".format(task_name, best_epoch) in res or "{}_{}_precision".format(task_name, best_epoch) in res:
        res["test_f1"] = float(res["{}_{}_f1".format(task_name, best_epoch)])
        res["test_precision"] = float(res["{}_{}_precision".format(task_name, best_epoch)])
    res["task_name"] = task_name
    # res["best_score"] = res["test_f1"]
    res["ZEN_model"] = res["bert_model"]
    res["path"] = eval_result_file

    return res

def stat_eval(args):
    ft_result_dict = defaultdict(dict)
    task_name_list = ["laptop", "rest14", "twitter"]
    for task_name in task_name_list:
        ft_result_dict[task_name] = defaultdict(dict)
    for task_name in task_name_list:
        print(task_name)
        eval_result_list = list(args.data_dir.glob("*{}*/eval_result*".format(task_name)))
        for eval_result in eval_result_list:
            try:
                res = get_eval_result(task_name, eval_result.parent)
                print(res)
            except Exception as e:
                print(str(e))
                print(eval_result.parent)
                continue

            res["task_name"] = task_name
            if res["ZEN_model"] not in ft_result_dict[task_name]:
                ft_result_dict[task_name][res["ZEN_model"]] = []
            ft_result_dict[task_name][res["ZEN_model"]].append(res)

    for task_name in task_name_list:
        print("#"*100)
        print(task_name)
        for ZEN_model, ZEN_model_data in ft_result_dict[task_name].items():
            print(ZEN_model)
            best_res = None
            for res in ZEN_model_data:
                if best_res is None or best_res["best_score"] < res["best_score"]:
                    best_res = res
            print("best:")
            res = best_res
            print("{} best_f1: {} acc: {} best_checkpoint: {} lr: {} warmup: {} train_batch_size: {} path: {}".format(
                task_name,
                res["test_f1"], res["test_precision"], res["best_checkpoint"], res["learning_rate"], res["warmup"],
                res["batch_size"], res["path"]))



def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=Path, required=True)
    args = parser.parse_args()

    stat_eval(args)


if __name__ == "__main__":
    main()
