# -*- coding: utf-8 -*-
import tools.config as config


def correlation_partition(tot_teacher_predicted_res, threshold=config.weak_threshold):
    tot_strong, tot_weak, tot_un = [], [], []
    for teacher_predicted_res in tot_teacher_predicted_res:
        strong_correlated = [i[0] % 50 for i in teacher_predicted_res[:6]]
        teacher_predicted_res = teacher_predicted_res[6:]
        weakly_correlated = [i[0] % 50 for i in teacher_predicted_res if i[1] >= threshold]
        uncorrelated = [i[0] % 50 for i in teacher_predicted_res if i[1] < threshold]
        tot_strong.append(strong_correlated)
        tot_weak.append(weakly_correlated)
        tot_un.append(uncorrelated)
    return tot_strong, tot_weak, tot_un


if __name__ == '__main__':
    pass
