#!/usr/bin/python

import codecs
import numpy as np
import os
import sys
import torch
import pandas as pd

from modules import BertNerData as NerData
from modules.models.bert_models import BertBiLSTMAttnCRF
from modules import NerLearner
from modules.data.bert_data import get_bert_data_loader_for_predict
from modules.train.train import validate_step
from modules.utils.plot_metrics import *


def read_data(input_file):
    """Reads a BIO data."""
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            
            if len(contends) == 0 and not len(words):
                words.append("")
            
            if len(contends) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label.replace("-", "_"))
        return lines


if __name__ == '__main__':
    print("Required Pytorch version is above 0.4.0, running with pytorch %s" % torch.__version__)
    print("usage: dataPath modelPath numEpochs")

    # Load data
    data_path = sys.argv[1]
    train_path = data_path + "train.txt"
    dev_path = data_path + "dev.txt"
    test_path = data_path + "test.txt"

    train_f = read_data(train_path)
    dev_f = read_data(dev_path)
    test_f = read_data(test_path)

    train_df = pd.DataFrame(train_f, columns=["0", "1"])
    train_df.to_csv(data_path + "train.csv", index=False)

    valid_df = pd.DataFrame(dev_f, columns=["0", "1"])
    valid_df.to_csv(data_path + "valid.csv", index=False)

    test_df = pd.DataFrame(test_f, columns=["0", "1"])
    test_df.to_csv(data_path + "test.csv", index=False)

    train_path = data_path + "train.csv"
    valid_path = data_path + "valid.csv"
    test_path = data_path + "test.csv"

    # Load model
    model_dir = sys.argv[2]
    init_checkpoint_pt = os.path.join(model_dir, "pytorch_model.bin")
    bert_config_file = os.path.join(model_dir, "bert_config.json")
    vocab_file = os.path.join(model_dir, "vocab.txt")

    torch.cuda.set_device(0)
    torch.cuda.is_available(), torch.cuda.current_device()

    data = NerData.create(train_path, valid_path, vocab_file)

    sup_labels = ['B_ORG', 'B_MISC', 'B_PER', 'I_PER', 'B_LOC', 'I_LOC', 'I_ORG', 'I_MISC']

    # create model
    model = BertBiLSTMAttnCRF.create(len(data.label2idx), bert_config_file, init_checkpoint_pt, enc_hidden_dim=256)
    model.get_n_trainable_params()

    # create learner
    num_epochs = int(sys.argv[3])
    learner = NerLearner(model, data,
                     best_model_path=os.path.join(os.environ["PHILLY_MODEL_DIRECTORY"], "bilstm_attn_cased.cpt"),
                     lr=0.001, clip=1.0, sup_labels=data.id2label[5:],
                     t_total=num_epochs * len(data.train_dl))

    # Start learning
    learner.fit(num_epochs, target_metric='f1')

    # Evaluate dev set
    dl = get_bert_data_loader_for_predict(data_path + "valid.csv", learner)

    learner.load_model()
    preds = learner.predict(dl)

    # IOB precision
    print(validate_step(learner.data.valid_dl, learner.model, learner.data.id2label, learner.sup_labels))

    # Span precision
    clf_report = get_bert_span_report(dl, preds, [])
    print(clf_report)

    # Evaluate test set
    dl = get_bert_data_loader_for_predict(data_path + "test.csv", learner)

    preds = learner.predict(dl)
    data = NerData.create(train_path, data_path + "test.csv", vocab_file)

    # IOB precision
    print(validate_step(data.valid_dl, learner.model, learner.data.id2label, learner.sup_labels))

    # Span precision
    clf_report = get_bert_span_report(dl, preds, [])
    print(clf_report)

    # Get mean and stdv on 10 runs
    num_runs = 10
    best_reports = []
    num_epochs = 100
    for i in range(num_runs):
        model = BertBiLSTMAttnCRF.create(len(data.label2idx), bert_config_file, init_checkpoint_pt, enc_hidden_dim=256)
        best_model_path = os.path.join(os.environ["PHILLY_MODEL_DIRECTORY"], "bilstm_attn_cased.cpt").format(i)
        learner = NerLearner(model, data,
                            best_model_path=best_model_path, verbose=False,
                            lr=0.001, clip=5.0, sup_labels=data.id2label[5:], t_total=num_epochs * len(data.train_dl))
        learner.fit(num_epochs, target_metric='f1')
        idx, res = get_mean_max_metric(learner.history, "f1", True)
        best_reports.append(learner.history[idx])

    #f1
    # Mean and std
    np.mean([get_mean_max_metric([r]) for r in best_reports]), np.round(np.std([get_mean_max_metric([r]) for r in best_reports]), 3)

    # Best
    get_mean_max_metric(best_reports)

    # #### precision
    # Mean and std
    np.mean([get_mean_max_metric([r], "prec") for r in best_reports]), np.round(np.std([get_mean_max_metric([r], "prec") for r in best_reports]), 3)

    # Best
    get_mean_max_metric(best_reports, "prec")