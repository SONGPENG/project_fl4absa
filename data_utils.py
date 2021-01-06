import os
import re
import logging
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, aspect, seg_list=[], label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.seg_list = seg_list
        self.aspect = aspect
        self.label = label

class DiaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir, tokenizer, max_seq_len=512, batch_size=16):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_size = 0

    def get_train_examples(self, dataset):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, dataset, "train.txt")), "train")

    def get_test_examples(self, dataset):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, dataset, "test.txt")), "test")

    def get_labels(self):
        return ["-1","0","1"]

    def get_label_map(self):
        label_map = {label: i for i, label in enumerate(self.get_labels())}
        return label_map

    def get_id2label_map(self):
        return {i: label for label, i in self.get_label_map().items()}

    def get_tag_size(self):
        return len(self.get_labels())
    
    def get_vocab_size(self):
        return self.vocab_size

    def _create_examples(self, lines, set_type):
        """
        examples = []
        for i, (sentence, seg_list, aspect, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text=sentence, aspect=aspect, seg_list=seg_list, label=label))
        """
        return lines

    def _read_txt(self, file_path):
        '''
        read file
        return format :
        '''
        '''
        datas = []
        sentence_list = []
        with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            for i, line in enumerate(fin):
                if i % 3 == 0:
                    datas.append([])
                datas[-1].append(line.strip())
            for data in datas:
                seg_list = re.split("\$T\$", data[0])
                aspect = data[1]
                polarity = data[2]
                sentence = aspect.join(seg_list)
                sentence_list.append([sentence, seg_list, aspect, polarity])
        '''
        sentence_list = []
        with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            for i, line in enumerate(fin):
                line = line.replace('!',' ').replace('@',' ').replace('(',' ').replace(')',' ').replace('{',' ').replace('}',' ').replace('[',' ').replace(']',' ').replace(';',' ').replace(':',' ').replace('"',' ').\
                replace('\'',' ').replace('?',' ').replace(',',' ').replace('...',' ')
                sentence_list.append(line)
        return sentence_list
    
    def verify_word(self, word):
        lower_case = []
        upper_case = []
        for i in range(26):
            lower_case.append(chr(97 + i))
            upper_case.append(chr(65 + i))
        for char in word:
            if char in lower_case or char in upper_case:
                return True
        return False
    
    def create_vocab(self, examples):
        vocab = set()
        vocab_dict = {}
        for example in examples:
            words = example.strip().split(' ')
            for word in words:
                if word == '':
                    continue
                if self.verify_word(word):
                    if vocab_dict.get(word, -1) == -1:
                        vocab_dict[word] = 0
                    vocab_dict[word] += 1
                    if vocab_dict[word] > 10:
                        vocab.add(word)
        return list(vocab)

    def convert_examples_to_features(self, tokenizer, examples):
        """Loads a data file into a list of `InputBatch`s."""
        vocab_file_path = os.path.join(os.getcwd(), 'data', 'generative_vocab.txt')
        if os.path.exists(vocab_file_path):
            f = open(vocab_file_path, 'r', encoding='utf-8')
            vocab = f.read().split('\n')
            f.close()
        else:
            vocab = self.create_vocab(examples)
            f = open(vocab_file_path, 'w', encoding='utf-8')
            for va in vocab:
                f.write(va + '\n')
            f.close()
        self.vocab_size = len(vocab)
        features = []
        word_2_id = {word:i for i, word in enumerate(vocab)}
        vocab_size = len(vocab)
        for (ex_index, example) in enumerate(examples):
            text = example
            tmp = [0 for i in range(vocab_size)]
            words = text.strip().split(' ')
            for word in words:
                if word == '':
                    continue
                if word in vocab:
                    tmp[word_2_id[word]] += 1
            features.append(tmp)
            
        return torch.tensor(features, dtype=torch.long)

    def _get_dataloader(self, features, batch_size, mode='train', rank=0, world_size=1):
        data_set = DiaDataset(features)
        sampler = RandomSampler(data_set)
        return DataLoader(data_set, sampler=sampler, batch_size=batch_size)

    def get_all_train_dataloader(self, dataset_list):
        tokenizer = self.tokenizer
        train_examples = []
        for dataset in dataset_list:
            train_examples.extend(self.get_train_examples(dataset))
        train_features = self.convert_examples_to_features(tokenizer, train_examples)
        train_dataloader = self._get_dataloader(train_features, mode="train", batch_size=self.batch_size)
        return train_dataloader

    def get_test_dataloader(self, dataset):
        tokenizer = self.tokenizer
        test_examples = self.get_test_examples(dataset)
        test_features = self.convert_examples_to_features(tokenizer, test_examples)
        test_dataloader = self._get_dataloader(test_features, mode="test", batch_size=self.batch_size)
        return test_dataloader

    def get_dataloader(self, dataset):
        tokenizer = self.tokenizer
        #train
        train_examples = self.get_train_examples(dataset)
        train_features = self.convert_examples_to_features(tokenizer, train_examples)
        train_dataloader = self._get_dataloader(train_features, mode="train", batch_size=self.batch_size)

        # test
        test_examples = self.get_test_examples(dataset)
        test_features = self.convert_examples_to_features(tokenizer, test_examples)
        test_dataloader = self._get_dataloader(test_features, mode="test", batch_size=self.batch_size)

        # dev
        dev_dataloader = test_dataloader

        return train_dataloader, test_dataloader, dev_dataloader

