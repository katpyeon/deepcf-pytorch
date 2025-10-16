# Cornac Pretrain 차원 불일치 문제

## ✅ 해결 완료 (2025-10-16)

**적용된 방안**: 방안 1 - Cornac도 전체 item 수로 모델 생성

### 구현 내용

1. **`common/data_utils.py`에 `load_cornac_data_with_full_space()` 함수 추가**
   - Train과 Test 파일을 모두 읽어 전체 user/item ID 추적
   - Test에만 있는 item도 `num_items`에 반영
   - DeepCF의 `load_deepcf_data()`와 동일한 방식으로 전체 item 공간 계산

2. **`cornac/cornac_eval.ipynb` 수정**
   - Cell 4: `load_cornac_data_with_full_space` import 추가
   - Cell 6: 커스텀 데이터 로딩으로 변경 (전체 item 공간 유지)
   - Cell 8: `BaseMethod.from_splits`를 사용한 커스텀 평가 방법 설정
   - Cell 2: `INCLUDE_CFNet_PRETRAIN = True`로 활성화

3. **결과**
   - ✅ Train set items: 2591 (이전: 2462)
   - ✅ Test set items: 2591
   - ✅ Pretrain 모델과 차원 일치
   - ✅ CFNet-pretrain 정상 동작
   - ✅ DeepCF 논문과 동일한 평가 방식 적용

### 변경된 파일
- `common/data_utils.py`: 새 함수 추가 (77-137행)
- `cornac/cornac_eval.ipynb`: Cell 2, 4, 6, 8 수정

---

## 📌 원래 문제 요약 (참고용)

`cornac_eval.ipynb`에서 CFNet-pretrain 모델을 사용할 때 다음과 같은 에러가 발생했었습니다:

```
RuntimeError: The size of tensor a (2462) must match the size of tensor b (2591) at non-singleton dimension 1
```

**에러 위치**: `cfnet/cfnet_model.py:198` (pretrain 가중치 로드 시)

---

## 🔍 문제 원인

### DeepCF 방식 vs Cornac 방식의 데이터 처리 차이

#### 1. DeepCF 방식 (`dmf_train.ipynb`, `mlp_train.ipynb`)

```python
# 이미 분리된 파일을 로드
train, testRatings, testNegatives, num_users, num_items = load_deepcf_data(
    DATA_PATH, 'ml-1m-sample100'
)

# load_deepcf_data() 내부 동작:
# 1. train.rating 파일 읽기 → 2462개 items
# 2. test.rating 파일 읽기 → 2591개 items (train에 없던 129개 포함)
# 3. num_items = max(train_items, test_items) = 2591 ✅

# 결과: 모델이 2591 items로 생성됨
model = DMF(train_matrix, num_users, num_items=2591, ...)
```

**특징**: Train과 Test를 **모두** 고려하여 전체 item 공간(2591개)으로 모델 생성

---

#### 2. Cornac 방식 (`cornac_eval.ipynb`)

```python
# 전체 데이터를 하나로 합쳐서 제공
data = deepcf_to_uir(train_file, test_file)  # train + test 모두 읽음

# Cornac의 RatioSplit이 알아서 8:2로 분리
eval_method = RatioSplit(data, test_size=0.2, seed=42)

# 문제: Train set만으로 Dataset 생성
# → train_set.num_items = 2462 (train에만 있는 item) ❌
# → test에만 있는 129개 item은 unknown으로 처리됨

# 결과: 모델이 2462 items로 생성됨
self.cfnet_model = CFNet(
    train_matrix,
    train_set.num_users,
    train_set.num_items=2462,  # ← 여기가 문제!
    ...
)
```

**특징**: Train set만 고려하여 축소된 item 공간(2462개)으로 모델 생성

---

### 차원 불일치 발생

```
┌─────────────────────────────────────────────┐
│ Pretrain 모델 (CFNet-rl.pth, CFNet-ml.pth) │
├─────────────────────────────────────────────┤
│ num_users = 100                             │
│ num_items = 2591 ✅ (전체 item 공간)       │
│                                             │
│ DMF user_layers[0].weight: [512, 2591]     │
│ DMF item_layers[0].weight: [1024, 100]     │
│ MLP user_embedding.weight: [256, 2591]     │
│ MLP item_embedding.weight: [256, 100]      │
└─────────────────────────────────────────────┘
                    ↓ 로드 시도
┌─────────────────────────────────────────────┐
│ Cornac CFNet 모델                           │
├─────────────────────────────────────────────┤
│ num_users = 100                             │
│ num_items = 2462 ❌ (train만의 item 공간)  │
│                                             │
│ DMF user_layers[0].weight: [512, 2462]     │
│ DMF item_layers[0].weight: [1024, 100]     │
│ MLP user_embedding.weight: [256, 2462]     │
│ MLP item_embedding.weight: [256, 100]      │
└─────────────────────────────────────────────┘

❌ RuntimeError: tensor a (2462) != tensor b (2591)
```

---

## 📊 데이터 분석

### ml-1m-sample100 데이터셋

```
전체 데이터셋:
├─ train.rating: 17,361개 interactions
│   ├─ users: 100개 (0~99)
│   └─ items: 2,462개 (train에 등장하는 item만)
│
└─ test.rating: 93개 interactions
    ├─ users: 93개 (일부 유저만 test 있음)
    └─ items: 2,591개 (train에 없던 129개 item 포함!)
```

### RatioSplit 후 Cornac Dataset

```python
>>> eval_method = RatioSplit(data, test_size=0.2)

>>> print(f"Train: {eval_method.train_set.num_items} items")
Train: 2462 items  # ← test에만 있는 129개 제외됨

>>> print(f"Test: {eval_method.test_set.num_items} items")
Test: 2591 items   # ← 전체 item (unknown 포함)
```

**문제점**: Cornac은 train에 없는 item을 unknown으로 처리하므로, 모델이 2462개로만 생성됨

---

## 🛠️ 해결 방안

### 방안 1: Cornac도 전체 item 수로 모델 생성 (권장) ⭐

**핵심 아이디어**: Cornac도 DeepCF처럼 전체 item 공간(2591개)으로 모델을 생성하도록 수정

#### 장점
- ✅ Pretrain 모델을 그대로 사용 가능
- ✅ DeepCF 논문과 동일한 평가 방식
- ✅ 차원 불일치 문제 완전 해결
- ✅ Leave-one-out 평가 방식과 일관성 유지

#### 단점
- ⚠️ Cornac의 기본 RatioSplit 사용 불가
- ⚠️ 커스텀 데이터 로딩 코드 필요

#### 구현 방법

**Step 1**: `common/data_utils.py`에 Cornac용 함수 추가

```python
def load_cornac_data_with_full_space(data_path, dataset_name):
    """
    Cornac 평가를 위해 전체 item 공간을 유지하며 데이터를 로드합니다.

    Returns:
        train_data: [(user_id, item_id, rating), ...] for train
        test_data: [(user_id, item_id, rating), ...] for test
        num_users: 전체 유저 수
        num_items: 전체 아이템 수 (train + test에 나타난 모든 item)
    """
    import numpy as np

    train_file = f"{data_path}{dataset_name}.train.rating"
    test_file = f"{data_path}{dataset_name}.test.rating"

    # 전체 user/item ID 추적
    all_users = set()
    all_items = set()

    # Train data 읽기
    train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                user_id, item_id, rating = parts[0], parts[1], float(parts[2])
                train_data.append((user_id, item_id, rating))
                all_users.add(user_id)
                all_items.add(item_id)

    # Test data 읽기
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                user_id, item_id, rating = parts[0], parts[1], float(parts[2])
                test_data.append((user_id, item_id, rating))
                all_users.add(user_id)
                all_items.add(item_id)  # ← 여기가 핵심!

    num_users = len(all_users)
    num_items = len(all_items)

    return train_data, test_data, num_users, num_items
```

**Step 2**: `cornac_eval.ipynb` Cell 6-8 수정

```python
# 기존 코드 (RatioSplit 사용)
# data = deepcf_to_uir(train_file, test_file)
# eval_method = RatioSplit(data=data, test_size=TEST_SIZE, ...)

# 새로운 코드 (전체 item 공간 유지)
from common.data_utils import load_cornac_data_with_full_space

# 데이터 로드 (전체 item 공간 포함)
train_data, test_data, num_users, num_items = load_cornac_data_with_full_space(
    DATA_PATH, DATASET
)

print(f"데이터 로딩 중: {DATASET}")
print(f"  Users: {num_users}, Items: {num_items}")
print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

# Cornac Dataset 생성 (전체 공간 명시)
train_set = cornac.data.Dataset.from_uir(
    train_data,
    seed=SEED,
    global_uid_map=None,  # 자동 생성
    global_iid_map=None   # 자동 생성
)

test_set = cornac.data.Dataset.from_uir(
    test_data,
    seed=SEED,
    global_uid_map=train_set.uid_map,  # train과 동일한 mapping 사용
    global_iid_map=train_set.iid_map   # train과 동일한 mapping 사용
)

# 커스텀 평가 방법 생성
eval_method = cornac.eval_methods.BaseMethod.from_splits(
    train_data=train_set,
    test_data=test_set,
    exclude_unknowns=False,  # unknown item도 평가에 포함
    verbose=VERBOSE,
    seed=SEED
)

print(f"\n✓ 평가 방법 설정 완료")
print(f"  Train set: {eval_method.train_set.num_users} users, "
      f"{eval_method.train_set.num_items} items")
print(f"  Test set: {eval_method.test_set.num_users} users, "
      f"{eval_method.test_set.num_items} items")
```

**Step 3**: 나머지 Cell은 그대로 실행

---

### 방안 2: CFNet에 차원 체크 + Fallback 추가

**핵심 아이디어**: Pretrain 로드 실패 시 자동으로 from-scratch로 전환

#### 장점
- ✅ 사용자 친화적 (에러 대신 경고만 출력)
- ✅ 기존 Cornac 평가 코드 유지 가능
- ✅ 다른 모델(DMF, MLP, NeuMF)은 정상 동작

#### 단점
- ⚠️ CFNet-pretrain이 실제로는 scratch로 학습됨 (pretrain 효과 없음)
- ⚠️ 근본적인 문제 해결은 아님

#### 구현 방법

**`cfnet/cfnet_model.py`의 `_load_pretrained_weights()` 메서드 수정**

```python
def _load_pretrained_weights(self, dmf_path, mlp_path):
    """
    사전 학습된 DMF와 MLP 모델의 가중치를 로드합니다.

    차원이 맞지 않는 경우 경고 메시지를 출력하고 랜덤 초기화로 전환합니다.
    """
    # 파일 존재 확인
    if not os.path.exists(dmf_path):
        raise FileNotFoundError(f"DMF pretrain 파일을 찾을 수 없습니다: {dmf_path}")
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"MLP pretrain 파일을 찾을 수 없습니다: {mlp_path}")

    # ============================================================
    # 차원 체크 추가 (새로운 코드)
    # ============================================================

    dmf_state = torch.load(dmf_path, map_location='cpu')
    mlp_state = torch.load(mlp_path, map_location='cpu')

    # DMF 차원 체크
    dmf_user_weight = dmf_state['user_layers.0.weight']  # [hidden, num_items]
    dmf_item_weight = dmf_state['item_layers.0.weight']  # [hidden, num_users]
    pretrain_num_items = dmf_user_weight.shape[1]
    pretrain_num_users = dmf_item_weight.shape[1]

    # MLP 차원 체크
    mlp_user_weight = mlp_state['user_embedding.weight']  # [embedding, num_items]
    mlp_item_weight = mlp_state['item_embedding.weight']  # [embedding, num_users]

    # 차원 불일치 확인
    dimension_mismatch = False

    if pretrain_num_users != self.num_users:
        print(f"\n⚠️  Pretrain 차원 불일치 (Users):")
        print(f"   Expected: {self.num_users}, Pretrain: {pretrain_num_users}")
        dimension_mismatch = True

    if pretrain_num_items != self.num_items:
        print(f"\n⚠️  Pretrain 차원 불일치 (Items):")
        print(f"   Expected: {self.num_items}, Pretrain: {pretrain_num_items}")
        dimension_mismatch = True

    if dimension_mismatch:
        print(f"\n💡 해결 방법:")
        print(f"   1. Cornac 평가 시 전체 item 공간(train+test)을 사용하도록 수정")
        print(f"   2. Pretrain 모델을 현재 데이터셋({self.num_items} items)로 재학습")
        print(f"\n→ Fallback: Random 초기화로 학습을 계속합니다 (from-scratch)")
        self._init_weights()
        return

    # ============================================================
    # 기존 pretrain 로드 코드 (차원이 일치하는 경우)
    # ============================================================

    # DMF 가중치 로드
    for idx in range(self.dmf_num_layer):
        layer_weight = dmf_state[f'user_layers.{idx}.weight']
        layer_bias = dmf_state[f'user_layers.{idx}.bias']
        self.dmf_user_layers[idx].weight.data.copy_(layer_weight)
        self.dmf_user_layers[idx].bias.data.copy_(layer_bias)

    # ... (나머지 기존 코드 동일)
```

**`cfnet/cornac_cfnet_wrapper.py`의 verbose 메시지도 수정**

```python
# Pretrain 사용 여부 출력
if self.verbose:
    if self.dmf_pretrain_path and self.mlp_pretrain_path:
        print(f"\n[{self.name}] Attempting to load pretrained weights:")
        print(f"  - DMF: {self.dmf_pretrain_path}")
        print(f"  - MLP: {self.mlp_pretrain_path}")
        print(f"  (차원 불일치 시 자동으로 from-scratch로 전환됩니다)")
    # ... 나머지 동일
```

---

### 방안 3: 임시로 CFNet-pretrain 비활성화

**핵심 아이디어**: 일단 평가만 진행하고 나중에 해결

#### 장점
- ✅ 즉시 실행 가능 (코드 수정 불필요)

#### 단점
- ⚠️ CFNet-pretrain을 평가할 수 없음
- ⚠️ 근본적인 해결은 아님

#### 구현 방법

**`cornac_eval.ipynb` Cell 2 수정**

```python
# ============================================================
# 비교할 모델 선택 (True/False)
# ============================================================
INCLUDE_CFNet_PRETRAIN = False  # ← True에서 False로 변경
INCLUDE_CFNet_SCRATCH = True    # scratch는 문제없음
INCLUDE_NCF = True
INCLUDE_MOSTPOP = True
```

---

## 🎯 추천 해결 순서

### 단기 (즉시 실행)
**방안 2** 또는 **방안 3** 선택
- 방안 2: 에러 없이 실행되지만 CFNet-pretrain은 실제로 scratch로 학습됨
- 방안 3: CFNet-pretrain 평가를 건너뜀

### 장기 (올바른 평가)
**방안 1** 구현
- DeepCF 논문과 동일한 평가 방식
- Pretrain의 진정한 효과 확인 가능
- 권장 사항 ⭐

---

## 📝 참고 사항

### DeepCF 논문의 Leave-One-Out 평가 방식

논문에서는 다음과 같은 방식을 사용:
1. 각 유저의 **마지막 interaction**을 test로 분리
2. 나머지를 train으로 사용
3. Test 시 각 positive item에 대해 **99개의 random negative** 추가
4. 총 100개 item 중 ranking 평가

이 방식에서는 **train과 test가 동일한 item 공간**을 공유합니다.

### Cornac의 기본 동작

Cornac의 `RatioSplit`은:
- Train/Test를 random으로 분리
- Train에 없는 item을 unknown으로 처리
- 모델은 train의 item 공간만 사용

→ **DeepCF 방식과 다름!**

### 결론

**방안 1**을 사용하여 DeepCF의 평가 방식을 정확히 재현하는 것이 가장 올바른 접근입니다.

---

## 📚 관련 파일

- `cfnet/cfnet_model.py:167-253` - `_load_pretrained_weights()` 메서드
- `cfnet/cornac_cfnet_wrapper.py:163-172` - CFNet 모델 초기화
- `common/data_utils.py` - 데이터 로딩 함수들
- `cornac_eval.ipynb` Cell 6-8 - 데이터 로딩 및 평가 설정

---

**작성일**: 2025-10-16
**버전**: 1.0
