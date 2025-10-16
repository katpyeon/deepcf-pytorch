"""
CFNet (Collaborative Filtering Network) 모델

이 모듈은 DeepCF (AAAI 2019) 논문의 CFNet 모델 구현을 포함합니다.
CFNet은 DMF(representation learning)와 MLP(metric learning)를 결합한 앙상블 모델로,
두 접근법의 장점을 모두 활용합니다.

Classes:
    CFNet: DMF + MLP 앙상블의 PyTorch 구현

Note:
    CFNet은 두 가지 학습 방식을 지원합니다:
    1. Pretrain (권장): DMF와 MLP를 먼저 개별 학습 후 가중치 로드
    2. From Scratch: 랜덤 초기화로 직접 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class CFNet(nn.Module):
    """
    Collaborative Filtering Network (CFNet) 모델.

    이 모델은 DeepCF 논문의 CFNet (fusion) 접근법을 구현합니다.
    DMF의 element-wise product와 MLP의 concatenation을 결합하여
    최고의 추천 성능을 달성합니다.

    구조:
        - DMF Part:
            - User Tower: num_items -> userlayers[0] -> ... -> userlayers[-1]
            - Item Tower: num_users -> itemlayers[0] -> ... -> itemlayers[-1]
            - DMF Vector: element-wise product of user and item latent vectors

        - MLP Part:
            - User Embedding: num_items -> layers[0]//2 (linear)
            - Item Embedding: num_users -> layers[0]//2 (linear)
            - MLP Vector: concatenate([user, item]) -> MLP layers

        - Fusion:
            - Predict Vector: concatenate([DMF Vector, MLP Vector])
            - Prediction: Linear -> Sigmoid

    Args:
        train_matrix (np.ndarray): 이진 상호작용 행렬 (num_users, num_items)
        num_users (int): 전체 유저 수
        num_items (int): 전체 아이템 수
        userlayers (list): DMF user tower의 레이어 크기 (예: [512, 64])
        itemlayers (list): DMF item tower의 레이어 크기 (예: [1024, 64])
        layers (list): MLP 레이어 크기 (예: [512, 256, 128, 64])
        dmf_pretrain_path (str, optional): DMF pretrain 모델 경로
        mlp_pretrain_path (str, optional): MLP pretrain 모델 경로

    Example:
        >>> # Pretrain 사용 (권장)
        >>> model = CFNet(train_matrix, 6040, 3706,
        ...               userlayers=[512, 64],
        ...               itemlayers=[1024, 64],
        ...               layers=[512, 256, 128, 64],
        ...               dmf_pretrain_path='../models/DMF.pth',
        ...               mlp_pretrain_path='../models/MLP.pth')
        >>>
        >>> # From scratch
        >>> model = CFNet(train_matrix, 6040, 3706,
        ...               userlayers=[512, 64],
        ...               itemlayers=[1024, 64],
        ...               layers=[512, 256, 128, 64])
    """

    def __init__(self, train_matrix, num_users, num_items,
                 userlayers, itemlayers, layers,
                 dmf_pretrain_path=None,
                 mlp_pretrain_path=None):
        super(CFNet, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.dmf_num_layer = len(userlayers)
        self.mlp_num_layer = len(layers)

        # Train matrix를 버퍼로 등록
        self.register_buffer('user_matrix', torch.FloatTensor(train_matrix))
        self.register_buffer('item_matrix', torch.FloatTensor(train_matrix.T))

        # ============================================================
        # DMF Part (원본 CFNet.py 64-74 라인)
        # ============================================================

        # User tower
        self.dmf_user_layers = nn.ModuleList()
        self.dmf_user_layers.append(nn.Linear(num_items, userlayers[0]))
        for idx in range(1, len(userlayers)):
            self.dmf_user_layers.append(nn.Linear(userlayers[idx-1], userlayers[idx]))

        # Item tower
        self.dmf_item_layers = nn.ModuleList()
        self.dmf_item_layers.append(nn.Linear(num_users, itemlayers[0]))
        for idx in range(1, len(itemlayers)):
            self.dmf_item_layers.append(nn.Linear(itemlayers[idx-1], itemlayers[idx]))

        # ============================================================
        # MLP Part (원본 CFNet.py 76-84 라인)
        # ============================================================

        # Embedding layers
        embedding_size = layers[0] // 2
        self.mlp_user_embedding = nn.Linear(num_items, embedding_size)
        self.mlp_item_embedding = nn.Linear(num_users, embedding_size)

        # MLP layers
        self.mlp_layers = nn.ModuleList()
        for idx in range(1, len(layers)):
            self.mlp_layers.append(nn.Linear(layers[idx-1], layers[idx]))

        # ============================================================
        # Prediction Layer (원본 CFNet.py 86-91 라인)
        # ============================================================

        # DMF output dimension: userlayers[-1]
        # MLP output dimension: layers[-1]
        # Concatenated dimension: userlayers[-1] + layers[-1]
        self.prediction = nn.Linear(userlayers[-1] + layers[-1], 1)

        # ============================================================
        # 가중치 초기화
        # ============================================================

        # Pretrain 모델 로드 또는 랜덤 초기화
        if dmf_pretrain_path is not None and mlp_pretrain_path is not None:
            self._load_pretrained_weights(dmf_pretrain_path, mlp_pretrain_path)
        else:
            self._init_weights()

    def _init_weights(self):
        """
        Lecun normal 초기화를 사용하여 가중치를 초기화합니다.
        Pretrain을 사용하지 않는 경우 (from scratch) 호출됩니다.
        """
        # DMF part
        for layer in self.dmf_user_layers:
            nn.init.normal_(layer.weight, std=np.sqrt(1.0 / layer.in_features))
            nn.init.zeros_(layer.bias)

        for layer in self.dmf_item_layers:
            nn.init.normal_(layer.weight, std=np.sqrt(1.0 / layer.in_features))
            nn.init.zeros_(layer.bias)

        # MLP part
        nn.init.normal_(self.mlp_user_embedding.weight,
                       std=np.sqrt(1.0 / self.mlp_user_embedding.in_features))
        nn.init.zeros_(self.mlp_user_embedding.bias)
        nn.init.normal_(self.mlp_item_embedding.weight,
                       std=np.sqrt(1.0 / self.mlp_item_embedding.in_features))
        nn.init.zeros_(self.mlp_item_embedding.bias)

        for layer in self.mlp_layers:
            nn.init.normal_(layer.weight, std=np.sqrt(1.0 / layer.in_features))
            nn.init.zeros_(layer.bias)

        # Prediction layer
        nn.init.normal_(self.prediction.weight, std=np.sqrt(1.0 / self.prediction.in_features))
        nn.init.zeros_(self.prediction.bias)

    def _load_pretrained_weights(self, dmf_path, mlp_path):
        """
        사전 학습된 DMF와 MLP 모델의 가중치를 로드합니다.

        원본 CFNet.py의 load_pretrain_model1 (105-124 라인)과
        load_pretrain_model2 (126-145 라인) 로직을 구현합니다.

        Args:
            dmf_path (str): DMF 모델 파일 경로 (.pth)
            mlp_path (str): MLP 모델 파일 경로 (.pth)

        Raises:
            FileNotFoundError: 모델 파일이 존재하지 않는 경우
            RuntimeError: 가중치 형태가 맞지 않는 경우
        """
        # 파일 존재 확인
        if not os.path.exists(dmf_path):
            raise FileNotFoundError(f"DMF pretrain 파일을 찾을 수 없습니다: {dmf_path}")
        if not os.path.exists(mlp_path):
            raise FileNotFoundError(f"MLP pretrain 파일을 찾을 수 없습니다: {mlp_path}")

        # ============================================================
        # DMF 가중치 로드 (원본 105-124 라인)
        # ============================================================

        dmf_state = torch.load(dmf_path, map_location='cpu')

        # User tower 가중치 복사
        for idx in range(self.dmf_num_layer):
            layer_weight = dmf_state[f'user_layers.{idx}.weight']
            layer_bias = dmf_state[f'user_layers.{idx}.bias']
            self.dmf_user_layers[idx].weight.data.copy_(layer_weight)
            self.dmf_user_layers[idx].bias.data.copy_(layer_bias)

        # Item tower 가중치 복사
        for idx in range(self.dmf_num_layer):
            layer_weight = dmf_state[f'item_layers.{idx}.weight']
            layer_bias = dmf_state[f'item_layers.{idx}.bias']
            self.dmf_item_layers[idx].weight.data.copy_(layer_weight)
            self.dmf_item_layers[idx].bias.data.copy_(layer_bias)

        # DMF prediction layer 가중치 (나중에 concatenate에 사용)
        dmf_pred_weight = dmf_state['prediction.weight']  # [1, dmf_dim]
        dmf_pred_bias = dmf_state['prediction.bias']      # [1]

        # ============================================================
        # MLP 가중치 로드 (원본 126-145 라인)
        # ============================================================

        mlp_state = torch.load(mlp_path, map_location='cpu')

        # Embedding layer 가중치 복사
        self.mlp_user_embedding.weight.data.copy_(mlp_state['user_embedding.weight'])
        self.mlp_user_embedding.bias.data.copy_(mlp_state['user_embedding.bias'])
        self.mlp_item_embedding.weight.data.copy_(mlp_state['item_embedding.weight'])
        self.mlp_item_embedding.bias.data.copy_(mlp_state['item_embedding.bias'])

        # MLP layer 가중치 복사
        for idx in range(self.mlp_num_layer - 1):
            layer_weight = mlp_state[f'mlp_layers.{idx}.weight']
            layer_bias = mlp_state[f'mlp_layers.{idx}.bias']
            self.mlp_layers[idx].weight.data.copy_(layer_weight)
            self.mlp_layers[idx].bias.data.copy_(layer_bias)

        # MLP prediction layer 가중치
        mlp_pred_weight = mlp_state['prediction.weight']  # [1, mlp_dim]
        mlp_pred_bias = mlp_state['prediction.bias']      # [1]

        # ============================================================
        # Prediction Layer 초기화 (원본 121-122, 138-144 라인)
        # ============================================================

        # DMF와 MLP의 prediction 가중치를 concatenate
        # 원본 121 라인: new_weights = np.concatenate((dmf_prediction[0], np.array([[0,]] * dmf_layers[-1])), axis=0)
        # 원본 141 라인: new_weights = np.concatenate((dmf_prediction[0][:mlp_layers[-1]], mlp_prediction[0]), axis=0)
        #
        # 의미: CFNet의 prediction은 [dmf_vector, mlp_vector]를 입력으로 받으므로,
        #       DMF 가중치와 MLP 가중치를 concatenate해야 함

        new_weight = torch.cat([dmf_pred_weight, mlp_pred_weight], dim=1)  # [1, dmf_dim + mlp_dim]

        # 원본 122, 142-144 라인: 0.5 means the contributions of MF and MLP are equal
        new_bias = 0.5 * (dmf_pred_bias + mlp_pred_bias)

        self.prediction.weight.data.copy_(0.5 * new_weight)
        self.prediction.bias.data.copy_(new_bias)

    def forward(self, user_input, item_input):
        """
        Forward pass. DMF와 MLP를 병렬로 실행 후 concatenate합니다.

        원본 CFNet.py의 get_model 함수 (49-96 라인) 로직을 구현합니다.

        Args:
            user_input (torch.LongTensor): 유저 인덱스 (batch_size,)
            item_input (torch.LongTensor): 아이템 인덱스 (batch_size,)

        Returns:
            torch.Tensor: 예측 점수 (batch_size, 1)
        """
        # Multi-hot encoding: 상호작용 벡터 추출 (원본 59-62 라인)
        user_rating = self.user_matrix[user_input]  # (batch_size, num_items)
        item_rating = self.item_matrix[item_input]  # (batch_size, num_users)

        # ============================================================
        # DMF Part (원본 64-74 라인)
        # ============================================================

        # User tower (첫 레이어는 linear, 나머지는 ReLU)
        dmf_user_latent = self.dmf_user_layers[0](user_rating)
        for idx in range(1, self.dmf_num_layer):
            dmf_user_latent = F.relu(self.dmf_user_layers[idx](dmf_user_latent))

        # Item tower (첫 레이어는 linear, 나머지는 ReLU)
        dmf_item_latent = self.dmf_item_layers[0](item_rating)
        for idx in range(1, self.dmf_num_layer):
            dmf_item_latent = F.relu(self.dmf_item_layers[idx](dmf_item_latent))

        # Element-wise product (원본 74 라인)
        dmf_vector = dmf_user_latent * dmf_item_latent  # (batch_size, userlayers[-1])

        # ============================================================
        # MLP Part (원본 76-84 라인)
        # ============================================================

        # Embedding (linear activation)
        mlp_user_latent = self.mlp_user_embedding(user_rating)
        mlp_item_latent = self.mlp_item_embedding(item_rating)

        # Concatenation (원본 81 라인)
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)

        # MLP layers (ReLU activation) (원본 82-84 라인)
        for idx in range(self.mlp_num_layer - 1):
            mlp_vector = F.relu(self.mlp_layers[idx](mlp_vector))  # (batch_size, layers[-1])

        # ============================================================
        # Fusion (원본 86-91 라인)
        # ============================================================

        # Concatenate DMF and MLP vectors (원본 87 라인)
        predict_vector = torch.cat([dmf_vector, mlp_vector], dim=-1)  # (batch_size, dmf_dim + mlp_dim)

        # Final prediction with sigmoid (원본 90-91 라인)
        prediction = torch.sigmoid(self.prediction(predict_vector))

        return prediction
