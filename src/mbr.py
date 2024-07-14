import numpy as np

from metrics import Metric
from typing import List

def mbr(metric: Metric, cands: List[str], source: List[str]):
    """
    Calculate MBR using a reference-based value metric.
    """
    if len(cands) == 1: return [1]

    # Create a pairwise list of candidates for evaluation
    candidate_matrix = []
    for i in range(len(cands)):
        for j in range(len(cands)):
            candidate_matrix += [(cands[i], cands[j])]
    assert len(candidate_matrix) == len(cands)**2

    # Score list of candidates
    sources    = [source for _ in candidate_matrix]
    targets    = [i for (i, j) in candidate_matrix]
    references = [[j] for (i, j) in candidate_matrix]

    candidate_matrix_scores = metric(src=sources, pred=targets, ref=references)

    # Restore MBR as a 2D matrix
    score_matrix = np.zeros((len(cands), len(cands)))
    cnt = 0
    for i in range(len(cands)):
        for j in range(len(cands)):
            score_matrix[i][j] = candidate_matrix_scores[cnt]
            cnt += 1

    return np.mean(score_matrix, axis=1).tolist(), score_matrix


def rerank(metric: Metric, cands: List[str], source: List[str]):
    """
    Score a list of candidates using reference-free re-ranking
    """
    if len(cands) == 1: return [1]
    
    sources = [source for _ in cands]
    targets = cands

    assert len(sources) == len(targets)

    candidate_scores = metric(src=sources, pred=targets)

    return candidate_scores, candidate_scores
