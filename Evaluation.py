from matplotlib.mlab import PCA
import numpy as np
import math


def pca_dim_reduction(input_data, target_dim):
    reduced_dataset = []
    pca_obj = PCA(np.array(input_data))
    projected_dataset = pca_obj.Y.tolist()
    for projected_data in projected_dataset:
        reduced_data = []  # one data point with reduced dim
        for col in range(0, target_dim):
            reduced_data.append(projected_data[col])
        reduced_dataset.append(reduced_data)
    return reduced_dataset


def get_trustworthiness(reduced_dataset, original_dataset, k):
    rank_sum = 0
    n = len(reduced_dataset)
    for i in range(0, n):
        curr_point = reduced_dataset[i]
        curr_point_neib_ranking = sorted(reduced_dataset, key=lambda l: np.linalg.norm(np.array(l) - np.array(curr_point)), reverse=False)
        curr_point_neib_ranking_mapping = dict()
        for i_2 in range(1, n):
            curr_point_neib_ranking_mapping[str(curr_point_neib_ranking[i_2])] = i_2
        original_curr_point = original_dataset[i]
        original_curr_point_neib_ranking = sorted(original_dataset, key=lambda l: np.linalg.norm(np.array(l) - np.array(original_curr_point)), reverse=False)
        original_curr_point_neib_ranking_mapping = dict()
        for i_2 in range(1, n):
            original_curr_point_neib_ranking_mapping[str(original_curr_point_neib_ranking[i_2])] = i_2
        for j in range(0, n):
            if i == j:
                continue
            # reduced_rank = get_rank(curr_point, reduced_dataset[j], reduced_dataset)
            # reduced_rank = curr_point_neib_ranking.index(reduced_dataset[j])
            reduced_rank = curr_point_neib_ranking_mapping[str(reduced_dataset[j])]
            # original_rank = get_rank(original_curr_point, original_dataset[j], original_dataset)
            # original_rank = original_curr_point_neib_ranking.index(original_dataset[j])
            original_rank = original_curr_point_neib_ranking_mapping[str(original_dataset[j])]
            if (reduced_rank <= k) and (original_rank > k):
                rank_sum += original_rank - k
    print(rank_sum)
    result = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * rank_sum
    return result


def get_continuity(reduced_dataset, original_dataset, k):
    rank_sum = 0
    n = len(reduced_dataset)
    for i in range(0, n):
        curr_point = reduced_dataset[i]
        curr_point_neib_ranking = sorted(reduced_dataset, key=lambda l: np.linalg.norm(np.array(l) - np.array(curr_point)), reverse=False)
        curr_point_neib_ranking_mapping = dict()
        for i_2 in range(1, n):
            curr_point_neib_ranking_mapping[str(curr_point_neib_ranking[i_2])] = i_2
        original_curr_point = original_dataset[i]
        original_curr_point_neib_ranking = sorted(original_dataset, key=lambda l: np.linalg.norm(np.array(l) - np.array(original_curr_point)), reverse=False)
        original_curr_point_neib_ranking_mapping = dict()
        for i_2 in range(1, n):
            original_curr_point_neib_ranking_mapping[str(original_curr_point_neib_ranking[i_2])] = i_2
        for j in range(0, n):
            if i == j:
                continue
            # reduced_rank = get_rank(curr_point, reduced_dataset[j], reduced_dataset)
            # reduced_rank = curr_point_neib_ranking.index(reduced_dataset[j])
            reduced_rank = curr_point_neib_ranking_mapping[str(reduced_dataset[j])]
            # original_rank = get_rank(original_curr_point, original_dataset[j], original_dataset)
            # original_rank = original_curr_point_neib_ranking.index(original_dataset[j])
            original_rank = original_curr_point_neib_ranking_mapping[str(original_dataset[j])]
            if (reduced_rank > k) and (original_rank <= k):
                rank_sum += reduced_rank - k
    print(rank_sum)
    result = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * rank_sum
    return result


# def get_generalization_error(reduced_dataset, original_dataset):
#     reduced_dataset_mapping = dict()
#     for i in range(0, len(reduced_dataset)):
#         reduced_dataset_mapping[str(reduced_dataset[i])] = i




# def get_trustworthiness(reduced_dataset, original_dataset, k):
#     rank_sum = 0
#     dists_between_low_dim_datapoints = []
#     n = len(reduced_dataset)
#     for i in range(0, n - 1):
#         for j in range(i + 1, n):
#             point_1 = reduced_dataset[i]
#             point_2 = reduced_dataset[j]
#             dist = np.linalg.norm(np.array(point_1) - np.array(point_2))
#             dists_between_low_dim_datapoints.append(dist)
#     sorted_dists = sorted(dists_between_low_dim_datapoints, key=float)
#     count = 0
#     for i in range(0, n):
#         curr_point = reduced_dataset[i]
#         curr_point_neib_ranking = sorted(reduced_dataset, key=lambda l: np.linalg.norm(np.array(l) - np.array(curr_point)), reverse=False)
#         curr_point_neib_ranking_mapping = dict()
#         for i_2 in range(1, n):
#             curr_point_neib_ranking_mapping[str(curr_point_neib_ranking[i_2])] = i_2
#         original_curr_point = original_dataset[i]
#         original_curr_point_neib_ranking = sorted(original_dataset, key=lambda l: np.linalg.norm(np.array(l) - np.array(original_curr_point)), reverse=False)
#         original_curr_point_neib_ranking_mapping = dict()
#         for i_2 in range(1, n):
#             original_curr_point_neib_ranking_mapping[str(original_curr_point_neib_ranking[i_2])] = i_2
#         for j in range(0, n):
#             if i == j:
#                 continue
#             # reduced_rank = get_rank(curr_point, reduced_dataset[j], reduced_dataset)
#             # reduced_rank = curr_point_neib_ranking.index(reduced_dataset[j])
#             reduced_rank = curr_point_neib_ranking_mapping[str(reduced_dataset[j])]
#             # original_rank = get_rank(original_curr_point, original_dataset[j], original_dataset)
#             # original_rank = original_curr_point_neib_ranking.index(original_dataset[j])
#             original_rank = original_curr_point_neib_ranking_mapping[str(original_dataset[j])]
#             if (reduced_rank <= k) and (original_rank > k):
#                 curr_dist = np.linalg.norm(np.array(curr_point) - np.array(reduced_dataset[j]))
#                 print(reduced_rank)
#                 count += 1
#                 rank_sum += sorted_dists.index(curr_dist) + 1 - k
#     print(count)
#     result = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * rank_sum
#     return result




def get_rank(datapoint_1, datapoint_2, dataset):    # what if two points have exactly the same coordinate?
    ranking = sorted(dataset, key=lambda l: np.linalg.norm(np.array(l) - np.array(datapoint_1)), reverse=False)
    ranking.remove(datapoint_1)
    for x in range(0, len(ranking)):
        if ranking[x] == datapoint_2:
            return x + 1
    return -1

