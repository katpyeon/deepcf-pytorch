# CFNet-PyTorch

DeepCF (AAAI 2019) 모델의 PyTorch 구현 (협업 필터링용)

**Original Implementation:** https://github.com/familyld/DeepCF

## ⚙️ 환경 설정

이 프로젝트는 **Anaconda** 가상환경을 사용합니다.

### 가상환경 생성 및 활성화

```bash
# 가상환경 생성 (Python 3.12)
conda create -n deepcf python=3.12

# 가상환경 활성화
conda activate deepcf

# 가상환경 비활성화 (작업 종료 시)
conda deactivate
```

### 필요한 패키지 설치

```bash
pip install -r requirement.txt
```

**주요 패키지:**
- PyTorch (torch, torchvision, torchaudio)
- NumPy, Pandas, scikit-learn
- Matplotlib
- Cornac (추천 시스템 평가 프레임워크)

## 📁 프로젝트 구조

```
deepcf-pytorch/
├── README.md                           # 이 파일
├── CLAUDE.md                           # Claude Code용 개발 가이드
├── requirement.txt                     # 필요한 패키지 목록
├── .gitignore                          # Git 제외 파일 목록
├── data_sampling.ipynb                 # 데이터 샘플링 전용 노트북
│
├── common/                             # 공통 유틸리티 (DMF, MLP, CFNet 공유)
│   ├── data_utils.py                   # 데이터 로딩 및 변환
│   ├── train_utils.py                  # 학습 관련 유틸리티
│   └── eval_utils.py                   # 평가 함수
│
├── cfnet_rl/                           # DMF (CFNet-rl) 모델
│   ├── dmf_model.py                    # DMF 모델 (PyTorch nn.Module)
│   ├── cornac_dmf_wrapper.py           # Cornac 래퍼
│   └── dmf_train.ipynb                 # DMF 독립 학습 노트북
│
├── cfnet_ml/                           # MLP (CFNet-ml) 모델
│   ├── mlp_model.py                    # MLP 모델 (PyTorch nn.Module)
│   ├── cornac_mlp_wrapper.py           # Cornac 래퍼
│   └── mlp_train.ipynb                 # MLP 독립 학습 노트북
│
├── cfnet/                              # CFNet (DMF+MLP fusion)
│   ├── cfnet_model.py                  # CFNet 모델 (PyTorch nn.Module)
│   ├── cornac_cfnet_wrapper.py         # Cornac 래퍼
│   ├── cfnet_train.ipynb               # CFNet 독립 학습 노트북 (pretrain/scratch)
│   └── PRETRAIN_LOGIC_DIFFERENCE.md    # Pretrain 로직 차이 문서
│
├── cornac/                             # Cornac 평가
│   ├── cornac_eval.ipynb               # 모든 모델 통합 평가 (베이스라인 비교)
│   ├── PRETRAIN_DIMENSION_ISSUE.md     # Pretrain 차원 불일치 이슈 문서
│   └── CornacExp-*.log                 # 평가 결과 로그 (자동 생성)
│
├── datasets/                           # DeepCF 포맷 데이터셋
│   ├── ml-1m.{train,test}.rating       # 전체 MovieLens-1M 데이터
│   └── ml-1m.test.negative             # 네거티브 샘플
│
└── pretrain/                           # 사전 학습된 모델 체크포인트 (.pth)
    ├── {dataset}-rl.pth                # DMF 모델 (직접 학습하여 생성)
    └── {dataset}-ml.pth                # MLP 모델 (직접 학습하여 생성)
```

> **참고:** `.pth` 파일은 용량이 크기 때문에 Git 저장소에 포함되지 않습니다. 아래 학습 가이드를 따라 직접 생성하세요.

## 🚀 빠른 시작

### 1️⃣ 데이터 및 샘플링된 데이터

`datasets/` 디렉토리에는 **전체 MovieLens-1M 데이터**와 **사용자 수로 미리 샘플링된 데이터**가 포함되어 있습니다.

**샘플 데이터셋:**
- `ml-1m-sample20`: 20명 유저 (빠른 테스트용) - 용량 문제로 포함하지 않음. 아래 샘플 생성 참고
- `ml-1m-sample100`: 100명 유저 (빠른 실험용) - 용량 문제로 포함하지 않음. 아래 샘플 생성 참고
- `ml-1m-sample500`: 500명 유저 (중간 규모) - 용량 문제로 포함하지 않음. 아래 샘플 생성 참고
- `ml-1m-sample1000`: 1000명 유저 (대규모 실험) - 용량 문제로 포함하지 않음. 아래 샘플 생성 참고
- `ml-1m`: 전체 6,040명 유저 (논문 재현용)

**추천:** 빠른 테스트와 실험을 위해 `ml-1m-sample100` 사용 
**추천:** 이후 전체 데이터 `ml-1m` 사용 

**샘플 생성이 필요한 경우:**
```bash
data_sampling.ipynb
```

---

### 2️⃣ Cornac 평가 

모든 모델을 **베이스라인과 비교 평가**합니다.

**파일:** `cornac/cornac_eval.ipynb`

**비교 모델:**
- **CFNet-rl** (DMF): Representation Learning
- **CFNet-ml** (MLP): Metric Learning
- **CFNet-scratch**: Fusion without pretrain
- **CFNet-pretrain**: Fusion with pretrain (최고 성능)
- **NeuMF**: Neural Collaborative Filtering 베이스라인
- **ItemPop**: 인기도 기반 베이스라인

**평가 지표:**
- **HR@10** (Hit Ratio): Top-10 추천 정확도
- **NDCG@10**: 순위 품질

**사용 방법:**
1. `cornac/cornac_eval.ipynb` 열기
2. **Cell 2**에서 `DATASET` 변수 수정:
   ```python
   DATASET = 'ml-1m'  # 샘플 크기 조정
   ```
3. 모든 셀 실행
4. 결과 테이블 확인 및 `cornac/CornacExp-*.log` 저장

---

### 3️⃣ 독립형 DMF 학습

DMF 모델을 **단독으로 학습**하고 평가합니다.

**파일:** `cfnet_rl/dmf_train.ipynb`

**주요 내용:**
- User/Item tower 기반 representation learning
- Element-wise product로 예측
- 모델 저장: `pretrain/{dataset_name}-rl.pth`

**사용 방법:**
1. `cfnet_rl/dmf_train.ipynb` 열기
2. **Cell 4**에서 `dataset_name` 수정:
   ```python
   dataset_name = 'ml-1m'  # 또는 'ml-1m-sample100'
   ```
3. 하이퍼파라미터 조정 (필요 시):
   ```python
   USERLAYERS = [512, 64]
   ITEMLAYERS = [1024, 64]
   EPOCHS = 20
   LEARNING_RATE = 0.0001
   ```
4. 모든 셀 실행
5. **결과:** `pretrain/ml-1m-rl.pth` 파일 생성 (약 200MB)

---

### 4️⃣ 독립형 MLP 학습

MLP 모델을 **단독으로 학습**하고 평가합니다.

**파일:** `cfnet_ml/mlp_train.ipynb`

**주요 내용:**
- User/Item embedding concatenation 기반 metric learning
- MLP로 상호작용 학습
- 모델 저장: `pretrain/{dataset_name}-ml.pth`

**사용 방법:**
1. `cfnet_ml/mlp_train.ipynb` 열기
2. **Cell 4**에서 `dataset_name` 수정:
   ```python
   dataset_name = 'ml-1m'  # 또는 'ml-1m-sample100'
   ```
3. 하이퍼파라미터 조정 (필요 시):
   ```python
   LAYERS = [512, 256, 128, 64]
   EPOCHS = 20
   LEARNING_RATE = 0.001
   ```
4. 모든 셀 실행
5. **결과:** `pretrain/ml-1m-ml.pth` 파일 생성 (약 180MB)

---

### 5️⃣ 독립형 CFNet 학습 (Fusion)

DMF와 MLP를 **결합한 CFNet**을 학습합니다.

**파일:** `cfnet/cfnet_train.ipynb`

**주요 내용:**
- DMF + MLP fusion 모델
- **Pretrain 모드**: DMF/MLP 가중치 로드 후 fine-tuning (최고 성능)
- **Scratch 모드**: 랜덤 초기화로 학습

**사용 방법:**
1. `cfnet/cfnet_train.ipynb` 열기
2. **Cell 4**에서 설정 수정:
   ```python
   dataset_name = 'ml-1m'  # 또는 'ml-1m-sample100'
   USE_PRETRAIN = True     # Pretrain 적용 여부
   ```
3. **Pretrain 사용 시 주의:**
   - 먼저 `cfnet_rl/dmf_train.ipynb` 실행 → `pretrain/{dataset}-rl.pth` 생성
   - 먼저 `cfnet_ml/mlp_train.ipynb` 실행 → `pretrain/{dataset}-ml.pth` 생성
   - 그 다음 CFNet 학습
4. 모든 셀 실행

**권장 워크플로우 (최고 성능):**
```
1. DMF 학습 (3️⃣) → pretrain/ml-1m-rl.pth (약 200MB)
2. MLP 학습 (4️⃣) → pretrain/ml-1m-ml.pth (약 180MB)
3. CFNet Pretrain 학습 (5️⃣) - 두 모델 결합
```

> **중요:** `.pth` 파일은 Git에 포함되지 않으므로, CFNet pretrain 모드를 사용하려면 반드시 DMF와 MLP를 먼저 학습해야 합니다.

## 📦 모델 아키텍처

### DMF (Deep Matrix Factorization) - CFNet-rl

**Representation Learning** 접근 방식

**아키텍처:**
- User Tower: Multi-hot encoding → [512, 64]
- Item Tower: Multi-hot encoding → [1024, 64]
- Prediction: Element-wise product → Sigmoid

**기본 하이퍼파라미터:**
```python
USERLAYERS = [512, 64]
ITEMLAYERS = [1024, 64]
EPOCHS = 20
BATCH_SIZE = 256
NUM_NEG = 4
LEARNING_RATE = 0.0001
LEARNER = 'adam'
```

### MLP (Multi-Layer Perceptron) - CFNet-ml

**Metric Learning** 접근 방식

**아키텍처:**
- User/Item Embedding → Concatenate → MLP [512, 256, 128, 64]
- Prediction: MLP output → Sigmoid

**기본 하이퍼파라미터:**
```python
LAYERS = [512, 256, 128, 64]
EPOCHS = 20
BATCH_SIZE = 256
NUM_NEG = 4
LEARNING_RATE = 0.001
LEARNER = 'adam'
```

### CFNet (Collaborative Filtering Net)

**DMF + MLP Fusion** 모델

**아키텍처:**
- DMF branch + MLP branch
- Concatenate outputs → Final prediction
- Pretrain 지원: DMF/MLP 가중치 로드

**학습 모드:**
- **Pretrain**: DMF/MLP 먼저 학습 후 fusion (최고 성능)
- **Scratch**: 랜덤 초기화로 학습

## 📊 데이터 포맷

이 프로젝트는 **DeepCF 포맷** (탭으로 구분)을 사용합니다.

**파일 구조:**
- `{dataset}.train.rating`: 학습 인터랙션
  ```
  userID\titemID\trating\ttimestamp
  ```
- `{dataset}.test.rating`: 테스트 인터랙션 (동일 포맷)
- `{dataset}.test.negative`: 테스트 네거티브 샘플 (leave-one-out 평가)
  ```
  (userID,itemID)\tnegativeItemID1\tnegativeItemID2\t...
  ```

**지원 데이터셋:**
- MovieLens-1M (ml-1m) 및 샘플 데이터셋

---

## 🎯 프로젝트 현황

### ✅ 완료된 기능
- [x] DMF 구현 (CFNet-rl)
- [x] MLP 구현 (CFNet-ml)
- [x] CFNet (DMF + MLP fusion)
- [x] Pretrain 지원 (DMF/MLP 가중치 로드)
- [x] Cornac 프레임워크 통합
- [x] 베이스라인 모델 비교 (NeuMF, ItemPop)
- [x] 샘플 데이터 생성 (20/100/500/1000 유저)
- [x] PyTorch 구현 검증 (원본 TensorFlow와 비교)

### 📝 알려진 이슈
- Pretrain 로직 차이 ([cfnet/PRETRAIN_LOGIC_DIFFERENCE.md](cfnet/PRETRAIN_LOGIC_DIFFERENCE.md) 참고)

## 📄 인용

```bibtex
@inproceedings{deng2019deepcf,
  title={DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System},
  author={Deng, Zhi-Hong and Huang, Ling and Wang, Chang-Dong and Lai, Jian-Huang and Philip, S Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={61--68},
  year={2019}
}
```

## 📝 라이선스

이 프로젝트는 학습 목적으로 제공됩니다.
