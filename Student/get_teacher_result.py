import pickle
import math

import numpy as np

import tools.config as config


def get_teacher_predicted_ranking(distance_type, data_name):
    all_res = []
    folder = f'./teacher_predict_result/{distance_type}/'
    source_distances = ['dtw', 'discret_frechet', 'hausdorff', 'erp']
    if distance_type in source_distances:
        source_distances.remove(distance_type)
    for source_distance in source_distances:
        res = []
        decoder = pickle.load(open(folder + f"{data_name}_{source_distance}_decoder_embeddings", 'rb'))[:3000]
        for st in range(0, 3000, 50):
            ed = st + 50
            embeddings = decoder[st:ed]
            for anchor_traj in range(50):
                test_distance = [(j + st, float(np.exp(-np.sum(np.square(embeddings[anchor_traj] - e)))))
                                 for j, e in enumerate(embeddings)]
                test_distance.sort(key=lambda x: x[1], reverse=True)
                res.append(test_distance)
        all_res.append(res)
    return all_res


def nDCG(predicted_rank, ideal_rank):
    dcg = 0
    idcg = 0
    for i in range(1, 6):
        fm = math.log(i + 2, 2)
        fz_dcg = math.exp(- 0.01 * ideal_rank.index(predicted_rank[i])) if predicted_rank[i] in ideal_rank else 0
        fz_idcg = math.exp(- 0.01 * i)
        dcg += fz_dcg / fm
        idcg += fz_idcg / fm
    return dcg / idcg


def R5IN15(predicted_rank, ideal_rank):
    cnt = 0
    for i in range(1, 16):
        if predicted_rank[i] in ideal_rank:
            cnt += 1
    return cnt / 5


def get_position_score(position, prob_rho=5):
    return math.e ** (-position / prob_rho)


def get_enriched_ranking_data(prob_rho, data_name, distance_type, all_predicted_rank):
    source_distance = ['discret_frechet', 'dtw', 'erp', 'hausdorff']
    if distance_type in source_distance:
        source_distance.remove(distance_type)
    distance_info_path = config.distance_info_path
    real_distance = pickle.load(open(distance_info_path, 'rb'))
    tot_score = []
    for anchor in real_distance:
        tot_ndcg = [0 for i in range(len(source_distance))]
        tot_recall = [0 for i in range(len(source_distance))]
        for source_idx, source_measure in enumerate(source_distance):
            predicted_rank = [x[0] for x in all_predicted_rank[source_idx][anchor]]
            real_rank = [y[0] for y in real_distance[anchor]]
            ndcg = nDCG(predicted_rank, real_rank)
            r5in15 = R5IN15(predicted_rank, real_rank)
            tot_ndcg[source_idx] += ndcg
            tot_recall[source_idx] += r5in15
        normalize_ndcg = [0 if sum(tot_ndcg) == 0 else i / sum(tot_ndcg) for i in tot_ndcg]
        normalize_recall = [0 if sum(tot_recall) == 0 else i / sum(tot_recall) for i in tot_recall]
        score = [normalize_ndcg[i] + normalize_recall[i] for i in range(len(normalize_recall))]
        normalize_score = [i / sum(score) for i in score]
        tot_score.append(normalize_score)
    tot_ideal_list = {}
    for anchor_idx, anchor in enumerate(real_distance):
        ideal_list = []
        real_rank = [y[0] for y in real_distance[anchor]]
        for target in range(anchor // 50 * 50, anchor // 50 * 50 + 50):
            prob_of_target = 0
            for source_idx, source_measure in enumerate(source_distance):
                if target in real_rank:
                    prob_of_target += ((6 - real_rank.index(target))) * tot_score[anchor_idx][source_idx]
                else:
                    predicted_rank = [x[0] for x in all_predicted_rank[source_idx][anchor]]
                    prob_of_target += tot_score[anchor_idx][source_idx] * get_position_score(
                        predicted_rank.index(target), prob_rho)
            ideal_list.append((target, prob_of_target))
        ideal_list = sorted(ideal_list, key=lambda x: x[1], reverse=True)
        tot_ideal_list[anchor] = ideal_list
    file_name = './teacher_predict_result/%s/%s_enriched_ranks_%d' % (distance_type, data_name, prob_rho)
    print(file_name)
    pickle.dump(tot_ideal_list, open(file_name, 'wb'))


if __name__ == '__main__':
    data_name = config.data_name
    distance_type = config.distance_type
    predicted_ranking = get_teacher_predicted_ranking(distance_type, data_name)
    get_enriched_ranking_data(config.prob_rho, data_name, distance_type, predicted_ranking)
