"""
평가 유틸리티

이 모듈은 leave-one-out 프로토콜을 사용한 추천 모델 평가 함수를 제공합니다.
이 유틸리티는 DMF, MLP, CFNet 모델에서 공유됩니다.

Functions:
    evaluate_model: leave-one-out 프로토콜을 사용한 모델 평가
"""

import numpy as np
import torch
import heapq
import math


def evaluate_model(model, testRatings, testNegatives, K, device):
    """
    leave-one-out 프로토콜을 사용하여 모델을 평가합니다.

    각 테스트 케이스에 대해, 99개의 부정 아이템 중에서 긍정 아이템의 순위를 매깁니다.
    Hit Ratio@K와 NDCG@K를 계산합니다.

    Args:
        model: PyTorch 모델 (DMF, MLP, 또는 CFNet)
        testRatings (list): [user, item] 테스트 쌍의 리스트
        testNegatives (list): 부정 아이템 리스트의 리스트 (테스트당 99개)
        K (int): 평가를 위한 Top-K
        device (torch.device): 계산을 위한 디바이스

    Returns:
        tuple: (hits, ndcgs) - 각 테스트 케이스에 대한 HR과 NDCG 리스트
    """
    hits, ndcgs = [], []

    model.eval()
    with torch.no_grad():
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            items = testNegatives[idx].copy()
            u = rating[0]
            gtItem = rating[1]  # ground truth item
            items.append(gtItem)

            # 예측 점수 계산
            users = np.full(len(items), u, dtype=np.int64)
            users_tensor = torch.LongTensor(users).to(device)
            items_tensor = torch.LongTensor(items).to(device)

            predictions = model(users_tensor, items_tensor).cpu().numpy().flatten()

            # 아이템-점수 매핑 생성
            map_item_score = {items[i]: predictions[i] for i in range(len(items))}
            items.pop()

            # Top-K 아이템 추출
            ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)

            # Hit Ratio 계산
            hr = 1 if gtItem in ranklist else 0
            hits.append(hr)

            # NDCG 계산
            ndcg = 0
            for i in range(len(ranklist)):
                if ranklist[i] == gtItem:
                    ndcg = math.log(2) / math.log(i + 2)
                    break
            ndcgs.append(ndcg)

    return (hits, ndcgs)
