import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from tqdm import tqdm

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Input L's labels & U's proba
class CategoryVectorInconsistencyAndRanking:
    """
    Uncertainty Sampling based on Category Vector Inconsistency and Ranking of Scores [RCV18]_
    selects instances based on the inconsistency of predicted labels and per-class label rankings.
    """

    def __init__(self, batch_size=2048, prediction_threshold=0.5, epsilon=1e-8, selected=torch.tensor([])):
        """
        Parameters
        ----------
        batch_size : int
            Batch size in which the computations are performed. Increasing the size increases
            the amount of memory used.
        prediction_threshold : float
            Confidence value above which a prediction counts as positive.
        epsilon : float
            A small value that is added to the argument of the logarithm to avoid taking the
            logarithm of zero.
        """
        self.batch_size = batch_size
        self.prediction_threshold = prediction_threshold
        self.epsilon = epsilon
        self.selected = np.array(selected)

    def query(self, y, pred, n=10):
        proba = np.array(pred)
        y = np.array(y)
        scores = self._compute_scores(y, proba)
        scores[self.selected] = 0

        # 此时返回 rank 前 n 的未标记数据的 index
        indices_queried = np.argpartition(-scores, n)[:n]
        return indices_queried
        # return np.array([indices_unlabeled[i] for i in indices_queried])

    def _compute_scores(self, y, proba):
        y_pred_unlabeled = (proba > self.prediction_threshold).astype(int)
        vector_inconsistency_scores = self._compute_vector_inconsistency(y,
                                                                         y_pred_unlabeled,
                                                                         proba.shape[1])
        ranking_scores = self._compute_ranking(proba)
        return vector_inconsistency_scores * ranking_scores

    def _compute_vector_inconsistency(self, y_arr, y_pred_unlabeled, num_classes):
        num_batches = int(np.ceil(len(y_pred_unlabeled) / self.batch_size))

        vector_inconsistency = np.array([], dtype=np.float32)
        num_unlabeled = y_pred_unlabeled.shape[0]

        # with build_pbar_context(self.pbar, tqdm_kwargs={'total': num_unlabeled}) as pbar:
        for batch_idx in np.array_split(np.arange(num_unlabeled), num_batches, axis=0):
            y_pred_unlabeled_sub = y_pred_unlabeled[batch_idx]
            # as an exception the variables a,b,c,d of the contingency table are adopted
            a = y_pred_unlabeled_sub.dot(y_arr.T)
            b = y_pred_unlabeled_sub.dot(-y_arr.T + 1)
            c = (-y_pred_unlabeled_sub + 1).dot(y_arr.T)
            d = (-y_pred_unlabeled_sub + 1).dot(-y_arr.T + 1)

            hamming_distance = (b + c) / num_classes

            distance = self._distance(y_pred_unlabeled_sub, y_arr, num_classes,
                                      a, b, c, d, hamming_distance)
            distance = distance.sum(axis=1) / y_arr.shape[0]
            vector_inconsistency = np.append(vector_inconsistency, distance)

        return vector_inconsistency

    def _distance(self, y_pred_unlabeled_sub, y_arr, num_classes, a, b, c, d,
                  hamming_distance):

        y_arr_ones = y_arr.sum(axis=1)
        y_arr_zeros = y_arr.shape[1] - y_arr_ones
        entropy_labeled = self._entropy(y_arr_ones, num_classes) \
                          + self._entropy(y_arr_zeros, num_classes)
        entropy_labeled = np.tile(entropy_labeled[np.newaxis, :],
                                  (y_pred_unlabeled_sub.shape[0], 1))

        y_pred_unlabeled_sub_ones = y_pred_unlabeled_sub.sum(axis=1)
        y_pred_unlabeled_sub_zeros = y_pred_unlabeled_sub.shape[1] - y_pred_unlabeled_sub_ones
        entropy_unlabeled = self._entropy(y_pred_unlabeled_sub_ones, num_classes) \
                            + self._entropy(y_pred_unlabeled_sub_zeros, num_classes)
        entropy_unlabeled = np.tile(entropy_unlabeled[:, np.newaxis], (1, y_arr.shape[0]))

        joint_entropy = self._entropy(b + c, num_classes) + self._entropy(a + d, num_classes)
        joint_entropy += (b + c) / num_classes \
                         * (self._entropy(b, b + c)
                            + self._entropy(c, b + c))
        joint_entropy += (a + d) / num_classes \
                         * (self._entropy(a, a + d) + self._entropy(d, a + d))

        entropy_distance = 2 * joint_entropy - entropy_unlabeled - entropy_labeled
        entropy_distance /= (joint_entropy + self.epsilon)

        distance = entropy_distance
        distance[hamming_distance == 1] = 1

        return distance

    def _entropy(self, numerator, denominator):
        ratio = numerator / (denominator + self.epsilon)
        result = -ratio * np.log2(ratio + self.epsilon)
        return result

    def _compute_ranking(self, proba_unlabeled):
        num_unlabeled, num_classes = proba_unlabeled.shape[0], proba_unlabeled.shape[1]
        ranks = self._rank_by_margin(proba_unlabeled)

        ranking_scores = [
            sum([num_unlabeled - ranks[j, i]
                 for j in range(num_classes)]) / (num_classes * (num_unlabeled - 1))
            for i in range(proba_unlabeled.shape[0])
        ]
        return np.array(ranking_scores)

    def _rank_by_margin(self, proba):
        num_classes = proba.shape[1]
        num_unlabeled = proba.shape[0]
        margin = np.abs(2 * proba - 1)

        ranks = np.zeros(shape=(num_classes, num_unlabeled))
        for i in range(num_classes):
            rank_index = np.argsort(margin[:, i])
            for j in range(num_unlabeled):
                ranks[i, rank_index[j]] = j + 1

        return ranks


class CategoryVectorInconsistencyAndRankingNew(CategoryVectorInconsistencyAndRanking):
    def __init__(self, data, batch_size=2048, prediction_threshold=0.5, epsilon=1e-8, selected=torch.tensor([]),
                 truth_mask=torch.tensor([])):
        super().__init__(batch_size=batch_size, prediction_threshold=prediction_threshold, epsilon=epsilon)
        self.truth_mask = np.array(truth_mask)
        self.selected = np.array(selected)
        self.data = data

    def query(self, y, pred, n=10):
        proba = np.array(pred)
        y = np.array(y)
        scores = self._compute_scores(y, proba)
        scores = scores * (1 - self.truth_mask)
        self._query_label_loc(scores, n)

        return self.selected, self.truth_mask

    def _compute_scores(self, y, proba):
        y_pred_unlabeled = (proba > self.prediction_threshold).astype(int)
        # filled y_pred_unlabeled with annotated labels
        # y_all = self.data['train_labels']
        # y_soft_unlabeled = np.array((y_all * self.truth_mask) + (y_pred_unlabeled * (1 - self.truth_mask)))
        vector_inconsistency_scores = self._compute_vector_inconsistency(y,
                                                                         y_pred_unlabeled,
                                                                         proba.shape[1])
        uncertain_scores = 2 * np.where(proba < 0.5, proba, 1 - proba)
        return (uncertain_scores.T * vector_inconsistency_scores).T

    def _query_label_loc(self, scores, label_size):
        num_classes = scores.shape[1]
        select_row = []
        label_col = []
        index = 1

        scores = torch.tensor(scores.reshape((-1,)))
        for i in range(label_size * num_classes):
            while len(select_row) == i:
                top_k = scores.topk(index).indices
                row, col = divmod(int(top_k[-1]), num_classes)
                index += 1
                # 当该条数据未被选择且对应的标注列未达到LABEL_SIZE时，选择此条数据此列进行标注
                # and update selected & mask
                if (row not in select_row) and label_col.count(col) < label_size:
                    select_row.append(row)
                    label_col.append(col)
                    self.truth_mask[row][col] = 1
                    if row not in self.selected:
                        self.selected = np.append(self.selected, row)


class MCVIE(CategoryVectorInconsistencyAndRanking):
    def __init__(self, data, data_dict, batch_size=2048, prediction_threshold=0.5, epsilon=1e-8, selected=torch.tensor([]),
                 truth_mask=torch.tensor([])):
        super().__init__(batch_size=batch_size, prediction_threshold=prediction_threshold, epsilon=epsilon)
        self.truth_mask = np.array(truth_mask)
        self.selected = np.array(selected)
        self.data = data
        self.data_dict = data_dict
        self.sentence_model = SentenceTransformer('./model/all-MiniLM-L6-v2')
        self.sentence_model.to(DEVICE)

    def query(self, y, pred, n=10):
        proba = np.array(pred)
        y = np.array(y)
        scores = self._compute_scores(y, proba)
        scores = scores * (1 - self.truth_mask)
        self._refine_ranking(scores, n)

        return self.selected, self.truth_mask

    def _compute_scores(self, y, proba):
        y_pred_unlabeled = (proba > self.prediction_threshold).astype(int)
        # filled y_pred_unlabeled with annotated labels
        y_all = self.data['train_labels']
        y_soft_unlabeled = np.array((y_all * self.truth_mask) + (y_pred_unlabeled * (1 - self.truth_mask)))
        vector_inconsistency_scores = self._compute_vector_inconsistency(y,
                                                                         y_soft_unlabeled,
                                                                         proba.shape[1])
        uncertain_scores = 2 * np.where(proba < 0.5, proba, 1 - proba)
        return (uncertain_scores.T * vector_inconsistency_scores).T

    def _refine_ranking(self, unc_1, label_size):
        # get embeddings
        sentences = self.data_dict["train"]['comment_text'].to_numpy()
        BATCH_SIZE = 64
        batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
        embeddings = []
        for batch in tqdm(batches):
            batch_embeddings = self.sentence_model.encode(batch, show_progress_bar=True)
            embeddings.append(torch.tensor(batch_embeddings))
        embeddings = torch.cat(embeddings)
        # select
        selected = self._label_once(unc_1)
        for _ in tqdm(range(label_size - 1)):
            center_embeddings = embeddings[selected].sum(dim=0) / len(selected)
            dist = torch.cdist(center_embeddings.unsqueeze(0), embeddings).squeeze(0) + self.epsilon
            r = torch.exp(-1 / dist)
            selected_tmp = self._label_once((unc_1 * np.array(r.cpu())).T)
            unc_1[selected_tmp] = 0
            selected = torch.cat([selected, selected_tmp])

    def _label_once(self, scores):
        num_classes = scores.shape[1]
        select_row = []
        label_col = []
        t_scores = torch.tensor(scores.reshape(-1))
        index = 1
        while len(select_row) < num_classes:
            top_k = t_scores.topk(index, largest=True)
            row, col = divmod(int(top_k.indices[-1]), num_classes)
            if (row not in select_row) and (col not in label_col):
                select_row.append(row)
                label_col.append(col)
                self.truth_mask[row][col] = 1
                # 将对应行的score置为0，不再被选取
                scores[row] = 0
                if row not in self.selected:
                    self.selected = np.append(self.selected, row)
            index += 1
        return torch.tensor(select_row, dtype=torch.long)

    def get_embedding(self, sentences):
        BATCH_SIZE = 32
        batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
        sentence_embeddings = []
        for batch in batches:
            batch_embeddings = self.sentence_model.encode(batch, show_progress_bar=True)
            sentence_embeddings.append(torch.tensor(batch_embeddings))
        sentence_embeddings = torch.cat(sentence_embeddings)
        return sentence_embeddings


class CBMAL:
    def __init__(self, data, batch_size=2048, prediction_threshold=0.5, epsilon=1e-8, selected=torch.tensor([]),
                 truth_mask=torch.tensor([])):
        self.data = data
        self.batch_size = batch_size
        self.prediction_threshold = prediction_threshold
        self.epsilon = epsilon
        self.truth_mask = np.array(truth_mask)
        self.selected = np.array(selected)

    def query(self, pred, n=10):
        # compute_unc
        uncertainty, w = self._compute_uncertainty(pred)
        # Refining the instance-label pairs
        self._refine_ranking(uncertainty, w, n)
        return self.selected, self.truth_mask

    def _compute_uncertainty(self, pred):
        # 1-uncertainty with pred
        unc_pred = 1 / (2 * abs(pred - 0.5) + self.epsilon)
        # 2-label correlation matrix
        class_num = pred.shape[1]
        y_part = self.truth_mask[self.selected] * np.array(self.data['train_labels'].iloc[self.selected, :])
        w = np.zeros((class_num, class_num), dtype=float)
        for i in range(class_num):
            tmp1 = y_part[:, i]
            sum_tmp1 = sum(tmp1 == 1)
            for j in range(class_num):
                if i == j:
                    continue
                tmp2 = y_part[:, i] + y_part[:, j]
                sum_tmp2 = sum(tmp2 == 2)
                w[i, j] = (sum_tmp2 + 1.0) / (max(sum_tmp1, 1) + 2.0)
        # 3-weights: w1、w2
        w1 = 1 - (w.sum(axis=1) / class_num)
        w2 = 1 - self.truth_mask.sum(axis=0) / pred.shape[0]
        return unc_pred * w1 * w2 * (1-self.truth_mask), w

    def _refine_ranking(self, unc_1, w, label_size):
        embeddings = self._compute_embeddings(self.data["train"]['comment_text'])
        embeddings = torch.Tensor(embeddings).cuda()
        # 第一次初始选取
        selected = self._label_once(unc_1)
        for i in range(label_size - 1):
            # Move the data to GPU
            selected = torch.LongTensor(selected).cuda()
            # Calculate the center embeddings on GPU
            center_embeddings = torch.sum(embeddings[selected], axis=0) / len(selected)
            # Initialize an empty tensor on the GPU
            dis = torch.empty(0).cuda()
            # Calculate the Euclidean distance for each embedding
            for k in range(embeddings.shape[0]):
                dis = torch.cat(
                    (dis, torch.tensor([distance.euclidean(embeddings[k], center_embeddings) + self.epsilon]).cuda()))
            # Move the result back to CPU
            dis = dis.cpu().numpy()
            # center_embeddings = embeddings[selected].sum(axis=0) / len(selected)
            # dis = np.array([])
            # for i in range(embeddings.shape[0]):
            #     dis = np.append(dis, distance.euclidean(embeddings[i], center_embeddings) + self.epsilon)
            r = np.exp(-1 / dis)
            d = len(selected) / (1 / np.exp(-w)).sum(axis=1)
            selected_tmp = self._label_once((unc_1.T * r).T * d)
            unc_1[selected_tmp] = 0
            selected = list(selected) + selected_tmp

    def _label_once(self, scores):
        num_classes = scores.shape[1]
        select_row = []
        label_col = []
        t_scores = torch.tensor(scores.reshape((-1,)))
        index = 1
        for i in range(num_classes):
            while len(select_row) == i:
                top_k = t_scores.topk(index).indices
                row, col = divmod(int(top_k[-1]), num_classes)
                index += 1
                if (row not in select_row) and label_col.count(col) < 1:
                    select_row.append(row)
                    label_col.append(col)
                    self.truth_mask[row][col] = 1
                    # 将对应行的score置为0，不再被选取
                    scores[row] = 0
                    if row not in self.selected:
                        self.selected = np.append(self.selected, row)
        return select_row

    def _compute_embeddings(self, text):
        model0 = SentenceTransformer('./model/distilbert-base-uncased')
        model0.to(DEVICE)
        sentence_embeddings = model0.encode(text)
        return sentence_embeddings


