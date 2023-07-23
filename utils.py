import pandas as pd
import random
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# Dir_path
dir_prefix_default = './data/jigsaw-toxic-comment-classification-challenge/sub/'


# Randomly split original data into train and test sets by 8:2 & 7:3
def split_original_data(dir_prefix=dir_prefix_default):
    data = pd.read_csv(dir_prefix + 'train.csv.zip')
    train, val = train_test_split(data, test_size=0.2, random_state=42)
    train, test = train_test_split(train, test_size=0.3, random_state=42)

    # Split data and labels
    train_data = train.copy()
    train_labels = train.drop('comment_text', axis=1)
    val_data = val.loc[:, ['id', 'comment_text']]
    val_labels = val.drop('comment_text', axis=1)
    test_data = test.loc[:, ['id', 'comment_text']]
    test_labels = test.drop('comment_text', axis=1)

    # Save to csv file
    train.to_csv(dir_prefix + 'train.csv', index=False)
    val_data.to_csv(dir_prefix + 'val.csv', index=False)
    val_labels.to_csv(dir_prefix + 'val_label.csv', index=False)
    test_data.to_csv(dir_prefix + 'test.csv',index=False)
    test_labels.to_csv(dir_prefix + 'test_label.csv',index=False)

    data_dict = {
        "train": train_data.reset_index(drop=True),
        "train_labels": train_labels.iloc[:, 1:].reset_index(drop=True),
        "val": val_data.reset_index(drop=True),
        "val_labels": val_labels.iloc[:, 1:].reset_index(drop=True),
        "test": test_data.reset_index(drop=True),
        "test_labels": test_labels.iloc[:, 1:].reset_index(drop=True)
    }

    return data_dict


# 20% part of original data
def part_data():
    dir_prefix = './jigsaw-toxic-comment-classification-challenge/'
    all = pd.read_csv(dir_prefix + 'train.csv.zip')
    part, remove = train_test_split(all, test_size=0.8, random_state=42)
    part.to_csv(dir_prefix + 'part/train.csv.zip', index=False)
    split_original_data(dir_prefix + 'part/')


# Resample original data to imbalance
def resample_data():
    data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv.zip')

    selected = []
    exclude = []
    label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for i in range(len(data)):
        if data[label_columns].loc[i].values.sum() != 0:
            selected.append(i)
        else:
            exclude.append(i)
    # save the toxic data
    toxic_data = data.loc[selected, :]
    toxic_data.to_csv('./jigsaw-toxic-comment-classification-challenge/resample/toxic.csv')
    print(len(exclude))
    random.seed(42)
    # add clean data
    selected_exc = random.sample(exclude, max(toxic_data.iloc[:, 2:8].sum().values))
    print(len(selected_exc))
    selected = sorted(selected + selected_exc)
    # formate resample data
    resample_data = data.loc[selected, :]
    resample_data.to_csv('./jigsaw-toxic-comment-classification-challenge/resample/train_r.csv', index=False)


def format_data(dir='./data/jigsaw-toxic-comment-classification-challenge/sub/'):
    train = pd.read_csv(dir + 'train.csv')
    train_labels = train.drop('comment_text', axis=1)
    val = pd.read_csv(dir + 'dev.csv')
    val_data = val.loc[:, ['id', 'comment_text']]
    val_labels = val.drop('comment_text', axis=1)
    test = pd.read_csv(dir + 'test.csv')
    test_data = test.loc[:, ['id', 'comment_text']]
    test_labels = test.drop('comment_text', axis=1)

    data_dict = {
        "train": train.reset_index(drop=True),
        "train_labels": train_labels.iloc[:, 1:].reset_index(drop=True),
        "val": val_data.reset_index(drop=True),
        "val_labels": val_labels.iloc[:, 1:].reset_index(drop=True),
        "test": test_data.reset_index(drop=True),
        "test_labels": test_labels.iloc[:, 1:].reset_index(drop=True)
    }

    return data_dict


# init_strategy from paper
def init_settings(df, label_names, init_size):
    new_label_count = {x: init_size for x in label_names}
    select_row = []
    truth_mask = [[0 for _ in range(len(label_names))] for _ in range(len(df))]

    for index, row in df.iterrows():
        labels = [l for l in label_names if row[l] == 1]
        add = False
        for l in labels:
            if new_label_count[l] > 0:
                add = True
                break
        if add == True:
            select_row.append(index)
            truth_mask[index] = [1 for _ in range(len(label_names))]
            for l in labels:
                new_label_count[l] -= 1
    # print(df.loc[select_row, :].iloc[:, 2:].sum().values)

    return torch.tensor(truth_mask), torch.tensor(select_row)


if __name__ == '__main__':
    # df = pd.read_csv(dir_prefix_default + 'part/train.csv')
    # label_names = df.columns[2:]
    # init_settings(df, label_names, 30)

    # part_data()
    split_original_data()

    # a = format_data()
    # print(1)