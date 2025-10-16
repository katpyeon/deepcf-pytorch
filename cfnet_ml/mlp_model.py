"""
Multi-Layer Perceptron (MLP) 모델

이 모듈은 DeepCF (AAAI 2019) 논문의 MLP 모델 구현을 포함합니다.
이 모델은 user-item 상호작용의 multi-hot encoding을 입력으로 사용하며,
user와 item embedding을 concatenate한 후 MLP로 학습합니다.

Classes:
    MLP: Multi-Layer Perceptron의 PyTorch 구현

Note:
    DMF와의 차이점:
    - DMF: element-wise product (곱셈)
    - MLP: concatenation (연결) + MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """
    Multi-Layer Perceptron 모델.

    이 모델은 DeepCF 논문의 CFNet-ml (metric learning) 접근법을 구현합니다.
    user-item 상호작용 벡터의 multi-hot encoding을 입력으로 사용하고,
    embedding을 concatenate한 후 MLP를 통해 학습합니다.

    구조:
        - User Embedding: num_items -> layers[0]//2 (linear)
        - Item Embedding: num_users -> layers[0]//2 (linear)
        - Concatenation: [user_latent, item_latent] -> layers[0]
        - MLP Layers: layers[1] -> layers[2] -> ... -> layers[-1] (ReLU)
        - Prediction: layers[-1] -> 1 (Sigmoid)

    Args:
        train_matrix (np.ndarray): 이진 상호작용 행렬 (num_users, num_items)
        num_users (int): 전체 유저 수
        num_items (int): 전체 아이템 수
        layers (list): MLP 레이어 크기 (예: [512, 256, 128, 64])
                      layers[0]은 user+item embedding 연결 후 크기
                      layers[0]//2가 각 embedding 크기

    Example:
        >>> from common.data_utils import get_train_matrix
        >>> train_matrix = get_train_matrix(train_sparse)
        >>> model = MLP(train_matrix, 6040, 3706, [512, 256, 128, 64])
        >>> predictions = model(user_indices, item_indices)
    """

    def __init__(self, train_matrix, num_users, num_items, layers):
        super(MLP, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = len(layers)

        # train matrix를 버퍼로 등록 (학습 가능한 파라미터 아님)
        # user_matrix[u] = 유저 u가 상호작용한 아이템들의 이진 벡터
        # item_matrix[i] = 아이템 i와 상호작용한 유저들의 이진 벡터
        self.register_buffer('user_matrix', torch.FloatTensor(train_matrix))
        self.register_buffer('item_matrix', torch.FloatTensor(train_matrix.T))

        # Embedding 레이어 (linear activation)
        # layers[0]//2: embedding 크기 (user와 item을 concat하면 layers[0])
        embedding_size = layers[0] // 2
        self.user_embedding = nn.Linear(num_items, embedding_size)
        self.item_embedding = nn.Linear(num_users, embedding_size)

        # MLP 레이어 (ReLU activation)
        self.mlp_layers = nn.ModuleList()
        for idx in range(1, len(layers)):
            self.mlp_layers.append(nn.Linear(layers[idx-1], layers[idx]))

        # Prediction 레이어
        self.prediction = nn.Linear(layers[-1], 1)

        # 가중치 초기화 (Lecun normal)
        self._init_weights()

    def _init_weights(self):
        """Lecun normal 초기화를 사용하여 가중치를 초기화합니다."""
        # Embedding layers
        nn.init.normal_(self.user_embedding.weight, std=np.sqrt(1.0 / self.user_embedding.in_features))
        nn.init.zeros_(self.user_embedding.bias)
        nn.init.normal_(self.item_embedding.weight, std=np.sqrt(1.0 / self.item_embedding.in_features))
        nn.init.zeros_(self.item_embedding.bias)

        # MLP layers
        for layer in self.mlp_layers:
            nn.init.normal_(layer.weight, std=np.sqrt(1.0 / layer.in_features))
            nn.init.zeros_(layer.bias)

        # Prediction layer
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

        # Embedding (linear activation)
        user_latent = self.user_embedding(user_rating)  # (batch_size, embedding_size)
        item_latent = self.item_embedding(item_rating)  # (batch_size, embedding_size)

        # Concatenation
        vector = torch.cat([user_latent, item_latent], dim=-1)  # (batch_size, layers[0])

        # MLP layers (ReLU activation)
        for layer in self.mlp_layers:
            vector = F.relu(layer(vector))

        # Sigmoid를 사용한 최종 예측
        prediction = torch.sigmoid(self.prediction(vector))

        return prediction
