"""
MLP를 위한 Cornac 프레임워크 래퍼

이 모듈은 MLP 모델을 위한 Cornac 호환 래퍼를 제공하여,
Cornac 추천 시스템 프레임워크와의 손쉬운 통합을 가능하게 합니다.

Classes:
    CornacMLP: MLP 모델을 위한 Cornac Recommender 래퍼

Example:
    >>> from mlp.cornac_mlp_wrapper import CornacMLP
    >>> mlp = CornacMLP(layers=[512, 256, 128, 64])
    >>> mlp.fit(train_set)
    >>> scores = mlp.score(user_idx=0)
"""

from cornac.models import Recommender
from cornac.exception import ScoreException
import torch
import numpy as np

# mlp 모듈과 공통 유틸리티에서 임포트
from .mlp_model import MLP
from common.data_utils import get_train_matrix
from common.train_utils import get_train_instances, TrainDataset
from torch.utils.data import DataLoader


class CornacMLP(Recommender):
    """
    Multi-Layer Perceptron (MLP)을 위한 Cornac Recommender 래퍼.

    이 클래스는 독립 실행형 MLP 구현을 Cornac 프레임워크와 함께 작동하도록 적응시켜,
    다른 추천 모델과의 자동 평가 및 비교를 가능하게 합니다.

    Args:
        name (str): 표시용 모델 이름
        layers (list): MLP 레이어 크기 (기본값: [512, 256, 128, 64])
                      layers[0]//2가 user/item embedding 크기
        num_epochs (int): 학습 에포크 수 (기본값: 20)
        batch_size (int): 학습 배치 크기 (기본값: 256)
        num_neg (int): 긍정 샘플당 부정 샘플 수 (기본값: 4)
        learning_rate (float): 학습률 (기본값: 0.001)
        learner (str): Optimizer 타입 ('adam', 'sgd', 'adagrad', 'rmsprop')
        use_gpu (bool): 가능한 경우 GPU 사용 (기본값: True)
        seed (int): 재현성을 위한 랜덤 시드 (기본값: None)
        verbose (bool): 학습 진행 상황 출력 (기본값: True)

    Example:
        >>> import cornac
        >>> from mlp.cornac_mlp_wrapper import CornacMLP
        >>>
        >>> # 모델 초기화
        >>> mlp = CornacMLP(
        ...     layers=[512, 256, 128, 64],
        ...     num_epochs=20,
        ...     use_gpu=True
        ... )
        >>>
        >>> # Cornac Experiment와 함께 사용
        >>> eval_method = cornac.eval_methods.RatioSplit(data, test_size=0.2)
        >>> cornac.Experiment(
        ...     eval_method=eval_method,
        ...     models=[mlp],
        ...     metrics=[cornac.metrics.NDCG(k=10)]
        ... ).run()
    """

    def __init__(
        self,
        name="MLP",
        layers=[512, 256, 128, 64],
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        learning_rate=0.001,
        learner='adam',
        use_gpu=True,
        seed=None,
        verbose=True
    ):
        Recommender.__init__(self, name=name, trainable=True, verbose=verbose)

        self.layers = layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.learning_rate = learning_rate
        self.learner = learner
        self.seed = seed

        # 디바이스 선택: CUDA > MPS > CPU
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    def fit(self, train_set, val_set=None):
        """
        Cornac train_set으로 MLP 모델을 학습합니다.

        Args:
            train_set (cornac.data.Dataset): 학습 데이터셋
            val_set (cornac.data.Dataset): 검증 데이터셋 (선택적, 미사용)

        Returns:
            self: 학습된 모델 인스턴스
        """
        Recommender.fit(self, train_set, val_set)

        # 재현성을 위한 랜덤 시드 설정
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Cornac train_set을 numpy 배열로 변환
        train_matrix_sparse = train_set.matrix.todok()
        train_matrix = get_train_matrix(train_matrix_sparse)

        # MLP 모델 초기화
        self.mlp_model = MLP(
            train_matrix,
            train_set.num_users,
            train_set.num_items,
            self.layers
        ).to(self.device)

        # Optimizer 설정
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.mlp_model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.mlp_model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.mlp_model.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=self.learning_rate)

        criterion = torch.nn.BCELoss()

        # 학습 루프
        for epoch in range(self.num_epochs):
            # 부정 샘플링
            user_input, item_input, labels = get_train_instances(
                train_matrix_sparse,
                self.num_neg,
                train_set.num_items
            )

            train_dataset = TrainDataset(user_input, item_input, labels)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            self.mlp_model.train()
            total_loss = 0

            for batch_users, batch_items, batch_labels in train_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)

                predictions = self.mlp_model(batch_users, batch_items)
                loss = criterion(predictions, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.verbose and epoch % 5 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f'  [MLP] Epoch {epoch:2d}: Loss = {avg_loss:.4f}')

        return self

    def score(self, user_idx, item_idx=None):
        """
        예측 점수를 계산합니다.

        Args:
            user_idx (int): 유저 인덱스
            item_idx (int or None): 아이템 인덱스. None이면 모든 아이템에 대해 점수 계산.

        Returns:
            float or np.ndarray: 예측 점수

        Raises:
            ScoreException: 유저 또는 아이템이 알려지지 않은 경우
        """
        self.mlp_model.eval()

        if item_idx is None:
            # 유저에 대한 모든 아이템 점수 계산
            if user_idx >= self.train_set.num_users:
                raise ScoreException(f"Unknown user: {user_idx}")

            item_indices = torch.arange(self.train_set.num_items, dtype=torch.long).to(self.device)
            user_indices = torch.full_like(item_indices, user_idx)

            with torch.no_grad():
                scores = self.mlp_model(user_indices, item_indices)

            return scores.cpu().numpy().flatten()
        else:
            # 특정 user-item 쌍에 대한 점수 계산
            if user_idx >= self.train_set.num_users:
                raise ScoreException(f"Unknown user: {user_idx}")
            if item_idx >= self.train_set.num_items:
                raise ScoreException(f"Unknown item: {item_idx}")

            with torch.no_grad():
                score = self.mlp_model(
                    torch.LongTensor([user_idx]).to(self.device),
                    torch.LongTensor([item_idx]).to(self.device)
                )

            return score.cpu().item()
