import os

import pandas as pd
import torch
import random

from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import distilBert
import numpy as np
import query_strategies
import utils

query_strategy = "mcvie"
# random、max_entropy、entropy_rate、cvirs、cvirs_new、cbmal、mcvie


# Config
# INIT_SIZE = 30
# LABEL_SIZE = 50
# TOP_K = 20
INIT_SIZE = 2
LABEL_SIZE = 3
CYCLES = 2


TOP_K = 3
P_threshold = 0.1
beta = 0.5

MAX_LEN = 512
train_params = {'batch_size': 16,
                'shuffle': True,
                'num_workers': 8
                }
val_params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 8
              }
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Load all train data
def init(strategy, data='toxic'):
    if data == 'toxic':
        data_dict = utils.split_original_data()
    else:
        data_dict = utils.format_data(data)
    train_data = data_dict['train']
    label_columns = train_data.columns[2:]
    if strategy == 'cvirs' or strategy == 'cvirs_new' or strategy == 'random' or strategy == 'max_entropy' or strategy == 'cbmal' or strategy == 'mcvie':
        truth_mask_global, select_row_global = utils.init_settings(data_dict['train'], label_columns, INIT_SIZE)
    else:
        truth_mask_global, select_row_global = init_settings(train_data[label_columns], label_columns, INIT_SIZE)
    # train_data_all = pd.read_csv(train_dir)
    # label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    # truth_mask_global, select_row_global = init_settings(train_data_all[label_columns], label_columns)
    data_dict['train']['labels'] = data_dict['train'][label_columns].apply(lambda x: list(x), axis=1)
    # data_dict['train'].drop(label_columns, inplace=True, axis=1)
    # data_dict['train'].drop(['id'], inplace=True, axis=1)

    return data_dict, truth_mask_global, select_row_global


# Init train_init、truth_mask
def init_settings(train_data, label_columns, init_size):
    # train_len = len(train_data)
    # truth_mask = [[0 for _ in range(6)] for _ in range(train_len)]
    # random.seed(15)
    # init_row = random.sample(range(0, train_len), 6 * INIT_SIZE)
    # for i in range(6 * INIT_SIZE):
    #     truth_mask[init_row[i]][i // INIT_SIZE] = 1
    # return torch.tensor(truth_mask), torch.tensor(init_row)
    train_len = len(train_data)
    candidate_row = {i: [] for i in label_columns}
    clean_row = []

    for index, row in train_data.iterrows():
        for col in label_columns:
            if row[col] == 1:
                candidate_row[col].append(index)
        if row.sum() == 0:
            clean_row.append(index)

    select_row = []
    # label positive samples
    truth_mask = [[0 for _ in range(6)] for _ in range(train_len)]
    for col in label_columns:
        selected = random.sample(candidate_row[col], init_size // 2)
        select_row += selected

    for i in range(len(select_row)):
        truth_mask[select_row[i]][i // (init_size // 2)] = 1

    # label negative samples
    selected_clean = random.sample(clean_row, int(init_size / 2) * 6)
    for i in range(len(selected_clean)):
        truth_mask[selected_clean[i]][i // (init_size // 2)] = 1

    return torch.tensor(truth_mask), torch.tensor(list(set(select_row + selected_clean)))


# Load training used data
class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, truth_mask, pseudo_mask, pseudo_label, max_len, new_data=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.comment_text
        self.new_data = new_data

        if not new_data:
            self.targets = self.data.labels
            self.truth_mask = truth_mask
            self.pseudo_mask = pseudo_mask
            self.pseudo_label = pseudo_label

        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        out = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

        if not self.new_data:
            out['targets'] = torch.tensor(self.targets[index], dtype=torch.float)
            out['truth_mask'] = self.truth_mask[index]
            out['pseudo_mask'] = self.pseudo_mask[index]
            out['pseudo_label'] = self.pseudo_label[index]

        return out


# 通过全量的truth_mask、p_mask获得标记行组成的truth_mask、p_mask
def get_selected_mask(truth_mask, pseudo_mask, pseudo_label, select_row):
    truth_mask_select = truth_mask[select_row, :]
    pseudo_mask_select = pseudo_mask[select_row, :]
    pseudo_label_select = pseudo_label[select_row, :]

    return truth_mask_select, pseudo_mask_select, pseudo_label_select


# Baseline1: Entropy sampling
def entropy_sampling(train_pred, train_label, truth_mask_global, pseudo_mask_global, select_row_global):
    entropy = -train_label * torch.log(train_pred) - (1 - train_label) * torch.log(1 - train_pred)
    entropy = entropy * (1 - truth_mask_global)
    truth_mask_global, select_row_global = query_label_loc(entropy, truth_mask_global, select_row_global)
    return truth_mask_global, pseudo_mask_global, select_row_global


# Baseline2: Random sampling
def random_sampling(truth_mask_global, pseudo_mask_global, select_row_global):
    train_len = len(truth_mask_global)
    exclude_row = []
    for col in range(truth_mask_global.shape[1]):
        candidate_row = [i for i in range(train_len)]
        # remove labeled position from candidate_row
        for row in range(train_len):
            if truth_mask_global[row, col] == 1:
                candidate_row.remove(row)
        # remove labeled row in this cycle from candidate_row
        for row in exclude_row:
            if row in candidate_row:
                candidate_row.remove(row)
        select_row = random.sample(candidate_row, LABEL_SIZE)
        for row in select_row:
            if row not in select_row_global:
                select_row_global = torch.cat((select_row_global, torch.tensor(row).reshape(-1)))
        truth_mask_global[select_row, col] = 1
        exclude_row += select_row

    return truth_mask_global, pseudo_mask_global, select_row_global


# Entropy rate sampling
def entropy_rate_sampling(
        train_pred,
        train_label,
        truth_mask_global,
        pseudo_mask_global,
        select_row_global,
        thr):
    entropy = -train_label * torch.log(train_pred) - (1 - train_label) * torch.log(1 - train_pred)
    pseudo_mask_global_new, pseudo_label_global = update_pseudo_mask(entropy, pseudo_mask_global, thr)
    entropy_m = entropy_add_pseudo(entropy, select_row_global, truth_mask_global, pseudo_mask_global_new,
                                   pseudo_mask_global)
    # entropy_m = entropy_mask_trans(entropy, truth_mask_global)
    entropy_r = entropy_rate_topK(entropy_m)
    select_row, label_col = query_label_loc(entropy_r, truth_mask_global, select_row_global)
    truth_mask_global, select_row_global = update_truth_mask(truth_mask_global, select_row_global, select_row,
                                                             label_col)
    pseudo_mask_global_new[truth_mask_global == 1] = 0
    return truth_mask_global, pseudo_mask_global_new, select_row_global, pseudo_label_global


# 更新pseudo_mask
def update_pseudo_mask(entropy, pseudo_mask, thr):
    cp = pseudo_mask.clone()
    pseudo_label = pseudo_mask.clone()
    # 只用负向的
    cp[entropy <= P_threshold] = 1

    pseudo_label[entropy > thr] = 1
    pseudo_label[entropy <= thr] = 0

    return cp, pseudo_label


# Use pseudo update entropy
def entropy_add_pseudo(
        entropy,
        select_row,
        truth_mask,
        pseudo_mask,
        pseudo_mask_old):
    entropy_exclude_truth = entropy_mask_trans(entropy, truth_mask)
    new_p = pseudo_mask.clone()
    # 对于已经选过并作为 pseudo 的行，更新前后伪标签并不会提供 entropy
    # for index in select_row:
    #     new_p[index] = new_p[index] - pseudo_mask_old[index]
    # new_p[new_p < 0] = 0

    entropy_pseudo_sum = (entropy_exclude_truth * pseudo_mask).sum(dim=1)
    trans_sum_pseudo = entropy_pseudo_sum.reshape(len(entropy_pseudo_sum), 1)
    for index in select_row:
        trans_sum_pseudo[index] = 0
        # trans_sum_pseudo[index] = (entropy_exclude_truth[index] * new_p[index]).sum(dim=0)

    return entropy_mask_trans(entropy_exclude_truth + beta * trans_sum_pseudo, truth_mask + pseudo_mask)


# truth_mask更新entropy
def entropy_mask_trans(entropy, truth_mask):
    cp = truth_mask.clone()
    cp[cp == 1] = 2
    cp[cp == 0] = 1
    cp[cp == 2] = 0

    return entropy * cp


# 使用TOP_K计算每列熵占比
def entropy_rate_topK(entropy_m):
    topk_col_sum = torch.tensor([])

    for i in range(6):
        topk_col_sum = torch.cat((topk_col_sum, entropy_m[:, i].topk(TOP_K).values.sum().reshape(-1)))
    entropy_r = entropy_m / topk_col_sum
    return entropy_r


# 根据entropy_m查询选择
def query_label_loc(entropy_m, truth_mask_global, selected_row_global):
    class_num = truth_mask_global.shape[1]
    select_row = []
    label_col = []
    index = 1

    entropy_m = entropy_m.reshape((-1,))
    for i in range(LABEL_SIZE * class_num):
        while len(select_row) == i:
            top_k = entropy_m.topk(index).indices
            row, col = divmod(int(top_k[-1]), class_num)
            index += 1
            # 当该条数据未被选择且对应的标注列未达到LABEL_SIZE时，选择此条数据此列进行标注
            if (row not in select_row) and label_col.count(col) < LABEL_SIZE:
                select_row.append(row)
                label_col.append(col)
                truth_mask_global[row][col] = 1
                if row not in selected_row_global:
                    selected_row_global = np.append(selected_row_global, row)
    return truth_mask_global, selected_row_global


# Update truth_mask
def update_truth_mask(truth_mask, select_row_global, select_row, label_col):
    for index in range(len(select_row)):
        i = select_row[index]
        j = label_col[index]
        truth_mask[i][j] = 1
        if i not in select_row_global:
            select_row_global = torch.cat((select_row_global, torch.tensor(i).reshape(-1)))
    return truth_mask, select_row_global


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_data_dict(data_dict, tokenizer):
    # Prepare train_data_all_loader
    train_data_set = MultiLabelDataset(data_dict['train'], tokenizer, [], [], [], MAX_LEN, new_data=True)
    train_data_loader = DataLoader(train_data_set, **val_params)

    # Prepare val_loader
    val_set = MultiLabelDataset(data_dict['val'], tokenizer, [], [], [], MAX_LEN, new_data=True)
    val_loader = DataLoader(val_set, **val_params)

    # Prepare test_loader
    test_set = MultiLabelDataset(data_dict['test'], tokenizer, [], [], [], MAX_LEN, new_data=True)
    test_loader = DataLoader(test_set, **val_params)

    out = {
        'train_loader': train_data_loader,
        'train_labels': data_dict['train_labels'],
        'val_loader': val_loader,
        'val_labels': data_dict['val_labels'],
        'test_loader': test_loader,
        'test_labels': data_dict['test_labels'],
    }

    return out


def promoting_msg(truth_mask, targets):
    labels = targets.copy()
    labels[np.array(truth_mask) == 0] = -1
    neg_num = []
    pos_num = []

    for i in range(labels.shape[1]):
        # calculate thr use of negative labels
        neg_num.append(sum(labels.iloc[:, i] == 0))
        # calculate thr use of positive labels
        pos_num.append(sum(labels.iloc[:, i] == 1))

    print('the num of used neg label:', neg_num)
    print('the num of used pos label:', pos_num)


# Main
if __name__ == '__main__':
    # set random seed
    seed_torch(42)
    # Initialize labeled datasets by randomly, using INIT_SIZE
    data_dict, truth_mask_global, select_row_global = init(query_strategy, 'toxic')
    class_num = truth_mask_global.shape[1]
    pseudo_mask_global = torch.zeros(size=(len(data_dict['train']), class_num))
    pseudo_label_global = torch.zeros(size=(len(data_dict['train']), class_num))

    tokenizer = DistilBertTokenizer.from_pretrained('./model/distilbert-base-uncased', truncation=True, do_lower_case=True)
    data_loaded = load_data_dict(data_dict, tokenizer)

    # F1_score_list for log
    micro_list = []
    macro_list = []
    select_len = []
    pseudo_len = []

    for cycle in range(CYCLES):
        print('\n-----------CYCLES:', cycle, '-------------')

        # Load training data
        training_data = data_dict['train'].loc[select_row_global, :]
        training_data = training_data.reset_index(drop=True)
        truth_mask, pseudo_mask, pseudo_label = get_selected_mask(truth_mask_global, pseudo_mask_global,
                                                                  pseudo_label_global, select_row_global)
        training_set = MultiLabelDataset(training_data, tokenizer, truth_mask, pseudo_mask, pseudo_label, MAX_LEN)
        training_loader = DataLoader(training_set, **train_params)

        # Print promoting msg
        promoting_msg(truth_mask, data_loaded['train_labels'].loc[select_row_global, :])
        select_len.append(len(select_row_global))
        print('Use the num of selected row: ', select_len)
        # pseudo_len.append(torch.sum(pseudo_mask).item())
        # print('Use the num of pseudo label: ', pseudo_len)

        # Train a model
        model = distilBert.DistilBERTClass(class_num=class_num)
        model.to(DEVICE)
        thr = distilBert.train(model, training_loader, data_loaded, cycle)

        # Load the best model
        model = distilBert.DistilBERTClass(class_num=class_num)
        model.load_state_dict(torch.load(
            os.path.join('./model/best_model' + str(cycle) + '.pt')))
        model.to(DEVICE)

        # Get model acc
        print('\n------Staring testing------')
        test_pred = distilBert.test(model, data_loaded['test_loader'])
        mi_score, ma_score = distilBert.calculate_acc(test_pred, data_loaded['test_labels'], thr, cycle)
        micro_list.append(mi_score)
        macro_list.append(ma_score)

        # Discard the last cycle
        if cycle == CYCLES - 1:
            break

        # AL process
        print('------AL sampling------')
        if query_strategy == 'random':
            truth_mask_global, pseudo_mask_global, select_row_global = random_sampling(
                truth_mask_global, pseudo_mask_global, select_row_global)
        elif query_strategy == 'entropy_rate':
            # Calculate entropy of train_data_all
            train_pred = distilBert.test(model, data_loaded['train_loader'])
            truth_mask_global, pseudo_mask_global, select_row_global, pseudo_label_global = entropy_rate_sampling(
                train_pred, torch.tensor(np.array(data_dict['train_labels'])), truth_mask_global, pseudo_mask_global, select_row_global, thr)
        elif query_strategy == 'max_entropy':
            train_pred = distilBert.test(model, data_loaded['train_loader'])
            truth_mask_global, pseudo_mask_global, select_row_global = entropy_sampling(
                train_pred, torch.tensor(np.array(data_dict['train_labels'])), truth_mask_global, pseudo_mask_global, select_row_global)
        elif query_strategy == 'cvirs':
            train_pred = distilBert.test(model, data_loaded['train_loader'])
            y = data_loaded['train_labels'].loc[select_row_global, :]
            cvirs = query_strategies.CategoryVectorInconsistencyAndRanking(selected=select_row_global)
            selected = cvirs.query(y=y, pred=train_pred, n=LABEL_SIZE)
            select_row_global = torch.cat((select_row_global, torch.tensor(selected)))
            truth_mask_global[selected] = 1
        elif query_strategy == 'cvirs_new':
            test_training_set = MultiLabelDataset(training_data, tokenizer, [], [], [], MAX_LEN, new_data=True)
            test_training_loader = DataLoader(test_training_set, **val_params)
            labeled_pred = distilBert.test(model, test_training_loader)
            y_soft = (np.array(labeled_pred) > thr).astype(int)
            # filled y_soft with annotated labels
            # y = data_loaded['train_labels'].loc[select_row_global, :]
            # y_soft = (y * np.array(truth_mask)) + (y_soft * (1 - np.array(truth_mask)))
            train_pred = distilBert.test(model, data_loaded['train_loader'])
            cvirs_new = query_strategies.CategoryVectorInconsistencyAndRankingNew(data_loaded, selected=select_row_global, truth_mask=truth_mask_global)
            select_row_global, truth_mask_global = cvirs_new.query(y=y_soft, pred=train_pred, n=LABEL_SIZE)
        elif query_strategy == 'cbmal':
            train_pred = distilBert.test(model, data_loaded['train_loader'])
            cbmal = query_strategies.CBMAL(data_dict, selected=select_row_global, truth_mask=truth_mask_global)
            select_row_global, truth_mask_global = cbmal.query(pred=train_pred, n=LABEL_SIZE)
        elif query_strategy == 'mcvie':
            test_training_set = MultiLabelDataset(training_data, tokenizer, [], [], [], MAX_LEN, new_data=True)
            test_training_loader = DataLoader(test_training_set, **val_params)
            labeled_pred = distilBert.test(model, test_training_loader)
            y_soft = (np.array(labeled_pred) > thr).astype(int)
            # filled y_soft with annotated labels
            y = data_loaded['train_labels'].loc[select_row_global, :]
            y_soft = (y * np.array(truth_mask)) + (y_soft * (1 - np.array(truth_mask)))
            train_pred = distilBert.test(model, data_loaded['train_loader'])
            mcvie = query_strategies.MCVIE(data_loaded, data_dict,
                                           selected=select_row_global,
                                           truth_mask=truth_mask_global,)
            select_row_global, truth_mask_global = mcvie.query(y=y_soft, pred=train_pred, n=LABEL_SIZE)

        # Print f1_score
        print('micro=', micro_list)
        print('macro=', macro_list)

    print('------  Finished  ------')
    print('micro=', micro_list)
    print('macro=', macro_list)
