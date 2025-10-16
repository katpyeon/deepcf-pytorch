"""
Deep Matrix Factorization (DMF) 모델

이 모듈은 DeepCF (AAAI 2019) 논문의 핵심 DMF 모델 구현을 포함합니다.
이 모델은 user-item 상호작용의 multi-hot encoding을 입력으로 사용하며,
유저와 아이템을 위한 두 개의 독립적인 deep tower를 사용합니다.

Classes:
    DMF: Deep Matrix Factorization의 PyTorch 구현

Note:
    get_train_matrix() 함수는 DMF, MLP, CFNet 모델 간 공유를 위해
    common/data_utils.py로 이동되었습니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DMF(nn.Module):
    """
    Deep Matrix Factorization 모델.

    이 모델은 DeepCF 논문의 CFNet-rl (representation learning) 접근법을 구현합니다.
    user-item 상호작용 벡터의 multi-hot encoding을 입력으로 사용하고,
    독립적인 deep tower를 통해 latent representation을 학습합니다.

    구조:
        - User Tower: num_items -> userlayers[0] -> ... -> userlayers[-1]
        - Item Tower: num_users -> itemlayers[0] -> ... -> itemlayers[-1]
        - Prediction: element-wise product -> Linear -> Sigmoid

    Args:
        train_matrix (np.ndarray): 이진 상호작용 행렬 (num_users, num_items)
        num_users (int): 전체 유저 수
        num_items (int): 전체 아이템 수
        userlayers (list): user tower의 레이어 크기 (예: [512, 64])
        itemlayers (list): item tower의 레이어 크기 (예: [1024, 64])

    Example:
        >>> from common.data_utils import get_train_matrix
        >>> train_matrix = get_train_matrix(train_sparse)
        >>> model = DMF(train_matrix, 6040, 3706, [512, 64], [1024, 64])
        >>> predictions = model(user_indices, item_indices)
    """

    def __init__(self, train_matrix, num_users, num_items, userlayers, itemlayers):
        super(DMF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = len(userlayers)

        # train matrix를 버퍼로 등록 (학습 가능한 파라미터 아님)
        # user_matrix[u] = 유저 u가 상호작용한 아이템들의 이진 벡터
        # item_matrix[i] = 아이템 i와 상호작용한 유저들의 이진 벡터
        self.register_buffer('user_matrix', torch.FloatTensor(train_matrix))
        self.register_buffer('item_matrix', torch.FloatTensor(train_matrix.T))

        # User tower 구성
        self.user_layers = nn.ModuleList()
        self.user_layers.append(nn.Linear(num_items, userlayers[0]))
        for idx in range(1, len(userlayers)):
            self.user_layers.append(nn.Linear(userlayers[idx-1], userlayers[idx]))

        # Item tower 구성
        self.item_layers = nn.ModuleList()
        self.item_layers.append(nn.Linear(num_users, itemlayers[0]))
        for idx in range(1, len(itemlayers)):
            self.item_layers.append(nn.Linear(itemlayers[idx-1], itemlayers[idx]))

        # Prediction 레이어
        self.prediction = nn.Linear(userlayers[-1], 1)

        # 가중치 초기화 (Lecun normal)
        self._init_weights()

    def _init_weights(self):
        """Lecun normal 초기화를 사용하여 가중치를 초기화합니다."""
        for layer in self.user_layers:
            nn.init.normal_(layer.weight, std=np.sqrt(1.0 / layer.in_features))
            nn.init.zeros_(layer.bias)

        for layer in self.item_layers:
            nn.init.normal_(layer.weight, std=np.sqrt(1.0 / layer.in_features))
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.prediction.weight, std=np.sqrt(1.0 / self.prediction.in_features))
        nn.init.zeros_(self.prediction.bias)

    def forward(self, user_input, item_input):
        """
        Forward pass.

        Args:
            user_input (torch.LongTensor): 유저 인덱스 (batch_size,)
            item_input (torch.LongTensor): 아이템 인덱스 (batch_size,)

        Returns:
            torch.Tensor: 예측 점수 (batch_size, 1)
        """
        # Multi-hot encoding: 상호작용 벡터 추출
        user_rating = self.user_matrix[user_input]  # (batch_size, num_items)
        item_rating = self.item_matrix[item_input]  # (batch_size, num_users)

        # User tower (첫 레이어는 활성화 없음, 나머지는 ReLU)
        user_latent = self.user_layers[0](user_rating)
        for idx in range(1, self.num_layer):
            user_latent = F.relu(self.user_layers[idx](user_latent))

        # Item tower (첫 레이어는 활성화 없음, 나머지는 ReLU)
        item_latent = self.item_layers[0](item_rating)
        for idx in range(1, self.num_layer):
            item_latent = F.relu(self.item_layers[idx](item_latent))

        # Element-wise product
        predict_vector = user_latent * item_latent

        # Sigmoid를 사용한 최종 예측
        prediction = torch.sigmoid(self.prediction(predict_vector))

        return prediction
