"""
CFNet을 위한 Cornac 프레임워크 래퍼

이 모듈은 CFNet (DMF + MLP fusion) 모델을 위한 Cornac 호환 래퍼를 제공하여,
Cornac 추천 시스템 프레임워크와의 손쉬운 통합을 가능하게 합니다.

Classes:
    CornacCFNet: CFNet 모델을 위한 Cornac Recommender 래퍼

Example:
    >>> from cfnet.cornac_cfnet_wrapper import CornacCFNet
    >>> cfnet = CornacCFNet(
    ...     userlayers=[512, 64],
    ...     itemlayers=[1024, 64],
    ...     layers=[512, 256, 128, 64],
    ...     dmf_pretrain_path='../pretrain/CFNet-rl.pth',
    ...     mlp_pretrain_path='../pretrain/CFNet-ml.pth'
    ... )
    >>> cfnet.fit(train_set)
    >>> scores = cfnet.score(user_idx=0)
"""

from cornac.models import Recommender
from cornac.exception import ScoreException
import torch
import numpy as np

# cfnet 모듈과 공통 유틸리티에서 임포트
from .cfnet_model import CFNet
from common.data_utils import get_train_matrix
from common.train_utils import get_train_instances, TrainDataset
from torch.utils.data import DataLoader


class CornacCFNet(Recommender):
    """
    CFNet (Collaborative Filtering Network)을 위한 Cornac Recommender 래퍼.

    이 클래스는 독립 실행형 CFNet 구현을 Cornac 프레임워크와 함께 작동하도록 적응시켜,
    다른 추천 모델과의 자동 평가 및 비교를 가능하게 합니다.

    CFNet은 DMF (representation learning)와 MLP (metric learning)를 결합한 앙상블 모델로,
    두 가지 학습 방식을 지원합니다:
    1. Pretrain (권장): DMF와 MLP를 먼저 개별 학습 후 가중치 로드
    2. From Scratch: 랜덤 초기화로 직접 학습

    Args:
        name (str): 표시용 모델 이름
        userlayers (list): DMF user tower의 레이어 크기 (기본값: [512, 64])
        itemlayers (list): DMF item tower의 레이어 크기 (기본값: [1024, 64])
        layers (list): MLP 레이어 크기 (기본값: [512, 256, 128, 64])
        dmf_pretrain_path (str, optional): DMF pretrain 모델 경로 (.pth)
        mlp_pretrain_path (str, optional): MLP pretrain 모델 경로 (.pth)
        num_epochs (int): 학습 에포크 수 (기본값: 20)
        batch_size (int): 학습 배치 크기 (기본값: 256)
        num_neg (int): 긍정 샘플당 부정 샘플 수 (기본값: 4)
        learning_rate (float): 학습률 (기본값: 0.0001)
        learner (str): Optimizer 타입 ('adam', 'sgd', 'adagrad', 'rmsprop')
        use_gpu (bool): 가능한 경우 GPU 사용 (기본값: True)
        seed (int): 재현성을 위한 랜덤 시드 (기본값: None)
        verbose (bool): 학습 진행 상황 출력 (기본값: True)

    Example:
        >>> import cornac
        >>> from cfnet.cornac_cfnet_wrapper import CornacCFNet
        >>>
        >>> # Pretrain 모델 사용 (권장)
        >>> cfnet_pretrain = CornacCFNet(
        ...     name="CFNet-pretrain",
        ...     userlayers=[512, 64],
        ...     itemlayers=[1024, 64],
        ...     layers=[512, 256, 128, 64],
        ...     dmf_pretrain_path='../pretrain/CFNet-rl.pth',
        ...     mlp_pretrain_path='../pretrain/CFNet-ml.pth',
        ...     num_epochs=20,
        ...     use_gpu=True
        ... )
        >>>
        >>> # From scratch (pretrain 없이)
        >>> cfnet_scratch = CornacCFNet(
        ...     name="CFNet-scratch",
        ...     userlayers=[512, 64],
        ...     itemlayers=[1024, 64],
        ...     layers=[512, 256, 128, 64],
        ...     num_epochs=20,
        ...     use_gpu=True
        ... )
        >>>
        >>> # Cornac Experiment와 함께 사용
        >>> eval_method = cornac.eval_methods.RatioSplit(data, test_size=0.2)
        >>> cornac.Experiment(
        ...     eval_method=eval_method,
        ...     models=[cfnet_pretrain, cfnet_scratch],
        ...     metrics=[cornac.metrics.NDCG(k=10)]
        ... ).run()
    """

    def __init__(
        self,
        name="CFNet",
        userlayers=[512, 64],
        itemlayers=[1024, 64],
        layers=[512, 256, 128, 64],
        dmf_pretrain_path=None,
        mlp_pretrain_path=None,
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        learning_rate=0.0001,
        learner='adam',
        use_gpu=True,
        seed=None,
        verbose=True
    ):
        Recommender.__init__(self, name=name, trainable=True, verbose=verbose)

        self.userlayers = userlayers
        self.itemlayers = itemlayers
        self.layers = layers
        self.dmf_pretrain_path = dmf_pretrain_path
        self.mlp_pretrain_path = mlp_pretrain_path
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
        Cornac train_set으로 CFNet 모델을 학습합니다.

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
        # train_set.matrix는 실제 train 데이터만 포함하므로,
        # train_set.num_items (전체 item 공간) 크기로 확장 필요
        train_matrix_sparse = train_set.matrix.todok()

        num_users = train_set.num_users

        # ============================================================
        # Pretrain 모델 사용 시: pretrained 모델의 차원을 사용
        # Pretrain 미사용 시: train_set.num_items 사용
        # ============================================================
        if self.dmf_pretrain_path is not None:
            # Pretrained 모델에서 실제 item 차원 추출
            dmf_state = torch.load(self.dmf_pretrain_path, map_location='cpu')
            pretrain_num_items = dmf_state['user_layers.0.weight'].shape[1]
            num_items = pretrain_num_items

            if self.verbose:
                print(f"\n[{self.name}] Using pretrained model dimensions:")
                print(f"  - Pretrained num_items: {pretrain_num_items}")
                print(f"  - Train set num_items: {train_set.num_items}")
                if pretrain_num_items != train_set.num_items:
                    print(f"  ⚠️  Dimension mismatch detected - using pretrained dimension ({pretrain_num_items})")
        else:
            # Pretrain 미사용 시 train_set의 차원 사용
            num_items = train_set.num_items

        # train_matrix 생성 (pretrain 차원 또는 train_set 차원 사용)
        train_matrix = np.zeros([num_users, num_items], dtype=np.int32)

        # 실제 train 데이터 복사
        for (u, i) in train_matrix_sparse.keys():
            train_matrix[u][i] = 1

        # CFNet 모델 초기화 (pretrain 경로가 있으면 자동 로드)
        self.cfnet_model = CFNet(
            train_matrix,
            num_users,
            num_items,  # pretrain 사용 시 pretrained 차원, 아니면 train_set 차원
            self.userlayers,
            self.itemlayers,
            self.layers,
            dmf_pretrain_path=self.dmf_pretrain_path,
            mlp_pretrain_path=self.mlp_pretrain_path
        ).to(self.device)

        # 모델의 item 차원을 인스턴스 변수로 저장 (score 함수에서 사용)
        self.model_num_items = num_items

        # Pretrain 사용 여부 출력
        if self.verbose:
            if self.dmf_pretrain_path and self.mlp_pretrain_path:
                print(f"\n[{self.name}] Using pretrained weights:")
                print(f"  - DMF: {self.dmf_pretrain_path}")
                print(f"  - MLP: {self.mlp_pretrain_path}")
            elif self.dmf_pretrain_path:
                print(f"\n[{self.name}] Using DMF pretrained weights only:")
                print(f"  - DMF: {self.dmf_pretrain_path}")
            elif self.mlp_pretrain_path:
                print(f"\n[{self.name}] Using MLP pretrained weights only:")
                print(f"  - MLP: {self.mlp_pretrain_path}")
            else:
                print(f"\n[{self.name}] Training from scratch (no pretrain)")

        # Optimizer 설정
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.cfnet_model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.cfnet_model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.cfnet_model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.cfnet_model.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.cfnet_model.parameters(), lr=self.learning_rate)

        criterion = torch.nn.BCELoss()

        if self.verbose:
            print(f"\n[{self.name}] Training started!")

        # 학습 루프
        for epoch in range(self.num_epochs):
            # 부정 샘플링
            user_input, item_input, labels = get_train_instances(
                train_matrix_sparse,
                self.num_neg,
                num_items  # 모델과 동일한 item 차원 사용
            )

            train_dataset = TrainDataset(user_input, item_input, labels)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            self.cfnet_model.train()
            total_loss = 0

            for batch_users, batch_items, batch_labels in train_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)

                predictions = self.cfnet_model(batch_users, batch_items)
                loss = criterion(predictions, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.verbose and epoch % 5 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f'  [{self.name}] Epoch {epoch:2d}: Loss = {avg_loss:.4f}')

        if self.verbose:
            print(f"\n[{self.name}] Evaluation started!")

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
        self.cfnet_model.eval()

        if item_idx is None:
            # 유저에 대한 모든 아이템 점수 계산
            if user_idx >= self.train_set.num_users:
                raise ScoreException(f"Unknown user: {user_idx}")

            item_indices = torch.arange(self.model_num_items, dtype=torch.long).to(self.device)
            user_indices = torch.full_like(item_indices, user_idx)

            with torch.no_grad():
                scores = self.cfnet_model(user_indices, item_indices)

            return scores.cpu().numpy().flatten()
        else:
            # 특정 user-item 쌍에 대한 점수 계산
            if user_idx >= self.train_set.num_users:
                raise ScoreException(f"Unknown user: {user_idx}")
            if item_idx >= self.model_num_items:
                raise ScoreException(f"Unknown item: {item_idx}")

            with torch.no_grad():
                score = self.cfnet_model(
                    torch.LongTensor([user_idx]).to(self.device),
                    torch.LongTensor([item_idx]).to(self.device)
                )

            return score.cpu().item()
