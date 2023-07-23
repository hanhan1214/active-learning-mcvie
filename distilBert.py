import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
from tensorboardX import SummaryWriter
import numpy as np


# Config
writer = SummaryWriter('./runs')
EPOCHS = 1
alpha = 0.25
LEARNING_RATE = 1e-05
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# distilBert model
class DistilBERTClass(torch.nn.Module):
    def __init__(self, class_num=6):
        super(DistilBERTClass, self).__init__()

        self.bert = DistilBertModel.from_pretrained('./model/distilbert-base-uncased')
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, class_num)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        out = hidden_state[:, 0]
        out = self.classifier(out)
        return out


# Train
def train(
        model,
        training_loader,
        data_loaded,
        cycle,
):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    model.train()
    global_step = 1
    best_fscore = 0
    fscore_list = []
    best_thr = 0
    loss_item = 1.0
    for epoch in range(EPOCHS):
        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.float)
            truth_mask = data['truth_mask'].to(DEVICE, dtype=torch.float)
            pseudo_mask = data['pseudo_mask'].to(DEVICE, dtype=torch.float)
            pseudo_label = data['pseudo_label'].to(DEVICE, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            # calculate loss
            mask_entropy_loss = MaskEntropyLoss()
            loss = mask_entropy_loss(torch.sigmoid(outputs), targets, truth_mask, pseudo_mask, pseudo_label)
            loss_item = loss.item()

            writer.add_scalar('train_loss_cycle' + str(cycle), loss.item(), global_step=global_step)
            print("loss:", loss.item(), " step:", global_step)
            global_step += 1
            loss.backward()
            optimizer.step()

        # Find the best checkpoint every epoch
        if loss_item < 1:
            print('\n------Staring validation------Epoch:', epoch, '-----------\n')
            val_pred = test(model, data_loaded['val_loader'])
            test_score = find_best_cut(val_pred, data_loaded['val_labels'])
            fscore_list.append(test_score)
            print(fscore_list)
            fscore = test_score["best_f"]
            thr = test_score["best_thr"]
            # Save a checkpoint
            if fscore > best_fscore:
                best_fscore = fscore
                best_thr = thr
                torch.save(model.state_dict(),
                           f'./model/best_model' + str(cycle) + '.pt')

    return best_thr


# Loss 计算
class MaskEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(MaskEntropyLoss, self).__init__()

    def forward(self, inputs, targets, truth_mask, pseudo_mask, pseudo_label):
        truth_num = torch.count_nonzero(truth_mask)
        pseudo_num = torch.count_nonzero(pseudo_mask)
        entropy = - targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        loss_m = entropy * truth_mask / truth_num

        if pseudo_num != 0:
            entropy_p = - pseudo_label * torch.log(inputs) - (1 - pseudo_label) * torch.log(1 - inputs)
            loss_p = (entropy_p * pseudo_mask) / pseudo_num
            loss_m = (1 - alpha) * loss_m + alpha * loss_p

        return torch.sum(loss_m)


# Test
def test(model, test_loader):
    all_test_pred = []
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            predict = torch.sigmoid(outputs)

            all_test_pred.append(predict)

    return torch.cat(all_test_pred).cpu()


# Find best f1
def find_best_cut(pred, target):
    f1_max = -1
    final_thr = 0.5
    # the grid from which thresholds are selected
    thrl = list(np.arange(0.25, 0.75, 0.05))
    # [0.25 0.3  0.35 0.4  0.45 0.5  0.55 0.6  0.65 0.7  0.75]
    for thr in thrl:
        pred_thr = (np.array(pred) > thr).astype(int)
        f1_thr = f1_score(target, pred_thr, average="macro")
        # print("target:", target)
        # print("pred: ", pred_thr)

        if f1_thr > f1_max:
            final_thr = thr
            f1_max = f1_thr

    return {"best_thr": final_thr, "best_f": f1_max}


# Calculate acc
def calculate_acc(pred, target, thr, cycle):
    class_num = pred.shape[1]
    origin_pred = pred.clone()
    # Using the best thr
    pred = (np.array(pred) > thr).astype(int)

    t_num = target.sum().values
    f_num = len(target) - t_num

    diff = pd.DataFrame(target.values - pred)

    true_num = diff[diff == 0].count().values
    fn = diff[diff == 1].count().values
    fp = diff[diff == -1].count().values
    tp = t_num - fn
    tn = f_num - fp

    acc = true_num / (t_num + f_num)
    precision = tp / (tp + fp)
    recall = tp / t_num
    specificity = tn / (tn + fp)

    # 计算AUC
    auc = []
    f1_list = []
    for i in range(class_num):
        y_labels = target.iloc[:, i:i + 1]
        y_pred = origin_pred[:, i:i + 1]
        y_pred_labels = pred[:, i:i + 1]
        auc.append(roc_auc_score(y_labels, y_pred))
        f1_list.append(f1_score(y_labels, y_pred_labels, average=None)[1])

    # 计算F1_score
    micro_f1 = f1_score(target, pred, average='micro')
    macro_f1 = f1_score(target, pred, average='macro')

    with open('acc_all_train.txt', 'a', encoding='utf-8') as f:
        f.write("------------ " + str(cycle) + ' ------------\n')
        f.write('t_num:' + str(t_num) + '\n')
        f.write('f_num:' + str(f_num) + '\n')
        f.write('true_num:' + str(true_num) + '\n')
        f.write('tp:' + str(tp) + '\n')
        f.write('tn:' + str(tn) + '\n')
        f.write('fn:' + str(fn) + '\n')
        f.write('fp:' + str(fp) + '\n')
        f.write('acc:' + str(acc) + str(acc.sum() / class_num) + '\n')
        f.write('precision:' + str(precision) + str(precision.sum() / class_num) + '\n')
        f.write('recall:' + str(recall) + str(recall.sum() / class_num) + '\n')
        f.write('specificity:' + str(specificity) + str(specificity.sum() / class_num) + '\n')
        f.write('auc:' + str(auc) + '\n')
        f.write('f1_list:' + str(f1_list) + '\n')
        f.write('micro_f1:' + str(micro_f1) + '\n')
        f.write('macro_f1:' + str(macro_f1) + '\n')
        f.write('thr:' + str(thr) + '\n')
    f.close()

    return micro_f1, macro_f1
