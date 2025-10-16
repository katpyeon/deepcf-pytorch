"""
학습 유틸리티

이 모듈은 부정 샘플링(negative sampling)을 통한 학습 인스턴스 생성 함수 및 클래스를 제공합니다.
이 유틸리티는 DMF, MLP, CFNet 모델에서 공유됩니다.

Functions:
    get_train_instances: 부정 샘플링을 통한 학습 인스턴스 생성

Classes:
    TrainDataset: 학습 데이터를 위한 PyTorch Dataset 래퍼
"""

import numpy as np
from torch.utils.data import Dataset


def get_train_instances(train, num_negatives, num_items):
    """
    부정 샘플링(negative sampling)을 통한 학습 인스턴스를 생성합니다.

    각 긍정(positive) (user, item) 쌍에 대해, 유저가 상호작용하지 않은 아이템 중
    num_negatives개의 부정(negative) 아이템을 샘플링합니다.

    Args:
        train (scipy.sparse.dok_matrix): 학습 상호작용 행렬
        num_negatives (int): 긍정 샘플당 부정 샘플 개수
        num_items (int): 전체 아이템 수

    Returns:
        tuple: (user_input, item_input, labels)
            - user_input: 유저 인덱스 리스트
            - item_input: 아이템 인덱스 리스트
            - labels: 이진 레이블 리스트 (1=긍정, 0=부정)
    """
    user_input, item_input, labels = [], [], []

    for (u, i) in train.keys():
        # 긍정 인스턴스
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        # 부정 인스턴스
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)

    return user_input, item_input, labels


class TrainDataset(Dataset):
    """
    학습 데이터를 위한 PyTorch Dataset 래퍼.

    Args:
        user_input (list): 유저 인덱스
        item_input (list): 아이템 인덱스
        labels (list): 이진 레이블
    """

    def __init__(self, user_input, item_input, labels):
        self.user_input = np.array(user_input, dtype=np.int64)
        self.item_input = np.array(item_input, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_input[idx], self.item_input[idx], self.labels[idx]
