# CFNet Pretrain Prediction Layer 초기화 로직 차이

## 📌 요약

**원본 TensorFlow**와 **현재 PyTorch 구현**의 CFNet pretrain 로드 시 prediction layer 초기화 방식이 다릅니다.

- **원본**: DMF와 MLP prediction 가중치를 복잡한 2단계 과정으로 결합
- **현재**: DMF와 MLP prediction 가중치를 단순 concatenate

**결론**: **현재 PyTorch 구현 (옵션 2) 유지 권장** ✅
- Forward pass는 100% 동일 (모델 아키텍처 정확)
- Pretrain 초기화는 학습 시작점의 차이일 뿐
- 현재 구현이 더 직관적이고 논리적
- 원본 로직의 비대칭성이 오히려 의문

---

## 🔍 상세 비교

### 1. 원본 TensorFlow 로직 (CFNet.py)

#### Step 1: DMF pretrain 로드 후 (105-124행)

```python
def load_pretrain_model1(model, dmf_model, dmf_layers):
    # DMF user/item layers 복사 (107-117행)
    # ... (생략)

    # Prediction weights 초기화 (119-123행)
    dmf_prediction = dmf_model.get_layer('prediction').get_weights()
    # dmf_prediction[0].shape = [dmf_layers[-1], 1]  예: [64, 1]
    # dmf_prediction[1].shape = [1]  (bias)

    new_weights = np.concatenate(
        (dmf_prediction[0], np.array([[0,]] * dmf_layers[-1])),
        axis=0
    )
    # 결과: [dmf_layers[-1] + dmf_layers[-1], 1] = [128, 1]
    #       앞 64개 = DMF prediction weights
    #       뒤 64개 = 0으로 초기화 (MLP 부분)

    new_b = dmf_prediction[1]
    model.get_layer('prediction').set_weights([new_weights, new_b])

    return model
```

**의문점**: 왜 `dmf_layers[-1]`개 만큼 0을 추가? → MLP output dim은 `layers[-1]`인데?

---

#### Step 2: MLP pretrain 로드 후 (126-145행)

```python
def load_pretrain_model2(model, mlp_model, mlp_layers):
    # MLP embedding/layers 복사 (128-136행)
    # ... (생략)

    # Prediction weights 재초기화 (138-144행)
    dmf_prediction = model.get_layer('prediction').get_weights()
    # 위 Step 1에서 설정된 [128, 1] 가중치

    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    # mlp_prediction[0].shape = [mlp_layers[-1], 1]  예: [64, 1]

    new_weights = np.concatenate(
        (dmf_prediction[0][:mlp_layers[-1]], mlp_prediction[0]),
        axis=0
    )
    # 결과: [mlp_layers[-1] + mlp_layers[-1], 1] = [64 + 64, 1] = [128, 1]
    #       앞 64개 = dmf_prediction[0]의 앞 64개 (DMF 가중치 일부)
    #       뒤 64개 = MLP prediction weights

    new_b = dmf_prediction[1] + mlp_prediction[1]
    # bias는 단순 합

    # 0.5 means the contributions of MF and MLP are equal (143행 주석)
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])

    return model
```

**핵심 로직**:
```python
dmf_prediction[0][:mlp_layers[-1]]  # DMF 가중치의 앞 mlp_layers[-1]개만 선택
```

---

### 2. 현재 PyTorch 구현 (cfnet_model.py 167-252행)

```python
def _load_pretrained_weights(self, dmf_path, mlp_path):
    # 파일 로드
    dmf_state = torch.load(dmf_path, map_location='cpu')
    mlp_state = torch.load(mlp_path, map_location='cpu')

    # DMF layers 복사 (194-210행)
    for idx in range(self.dmf_num_layer):
        # user_layers, item_layers 복사
        # ...

    # DMF prediction layer 가중치 추출
    dmf_pred_weight = dmf_state['prediction.weight']  # [1, dmf_dim]
    dmf_pred_bias = dmf_state['prediction.bias']      # [1]

    # MLP layers 복사 (218-229행)
    # user_embedding, item_embedding, mlp_layers 복사
    # ...

    # MLP prediction layer 가중치 추출
    mlp_pred_weight = mlp_state['prediction.weight']  # [1, mlp_dim]
    mlp_pred_bias = mlp_state['prediction.bias']      # [1]

    # Prediction layer 초기화 (246-252행)
    new_weight = torch.cat([dmf_pred_weight, mlp_pred_weight], dim=1)
    # [1, dmf_dim + mlp_dim] = [1, 128]

    new_bias = 0.5 * (dmf_pred_bias + mlp_pred_bias)

    self.prediction.weight.data.copy_(0.5 * new_weight)
    self.prediction.bias.data.copy_(new_bias)
```

**핵심 로직**:
```python
torch.cat([dmf_pred_weight, mlp_pred_weight], dim=1)  # 단순 concatenate
```

---

## 📊 차이점 분석

### Shape 비교

**가정**: `userlayers=[512, 64]`, `itemlayers=[1024, 64]`, `layers=[512, 256, 128, 64]`

| 단계 | 원본 TensorFlow | 현재 PyTorch |
|-----|----------------|--------------|
| **DMF output dim** | 64 (userlayers[-1]) | 64 |
| **MLP output dim** | 64 (layers[-1]) | 64 |
| **CFNet prediction input** | 128 (DMF 64 + MLP 64) | 128 (DMF 64 + MLP 64) |
| | | |
| **DMF prediction weights** | [64, 1] | [1, 64] (transpose) |
| **MLP prediction weights** | [64, 1] | [1, 64] (transpose) |
| | | |
| **Step 1 (DMF 로드 후)** | [128, 1] = [64(DMF) + 64(zeros)] | - |
| **Step 2 (MLP 로드 후)** | [128, 1] = [64(DMF[:64]) + 64(MLP)] | [1, 128] = [64(DMF) + 64(MLP)] |
| | | |
| **최종 scaling** | 0.5 * weights | 0.5 * weights |
| **최종 bias** | 0.5 * (DMF bias + MLP bias) | 0.5 * (DMF bias + MLP bias) |

---

### 가중치 초기화 값 비교

**원본 TensorFlow**:
```
Prediction weight = 0.5 * [
    dmf_weights[0:64],   ← DMF prediction의 앞 64개
    mlp_weights[0:64]    ← MLP prediction 전체
]
```

**현재 PyTorch**:
```
Prediction weight = 0.5 * [
    dmf_weights[0:64],   ← DMF prediction 전체
    mlp_weights[0:64]    ← MLP prediction 전체
]
```

**차이점**: 없음! (dmf_layers[-1] == mlp_layers[-1] == 64인 경우)

---

### ⚠️ 원본 로직의 문제점

#### 1. 비대칭적 초기화

```python
# Step 1: dmf_layers[-1]개 만큼 0 추가
new_weights = np.concatenate(
    (dmf_prediction[0], np.array([[0,]] * dmf_layers[-1])),  # ← dmf_layers[-1]
    axis=0
)

# Step 2: mlp_layers[-1]개만 선택
new_weights = np.concatenate(
    (dmf_prediction[0][:mlp_layers[-1]], mlp_prediction[0]),  # ← mlp_layers[-1]
    axis=0
)
```

**의문**:
- Step 1에서는 `dmf_layers[-1]`를 사용
- Step 2에서는 `mlp_layers[-1]`를 사용
- 두 값이 다르면? (예: dmf=64, mlp=128)

#### 2. 실제 CFNet prediction input dimension

CFNet forward pass (CFNet.py 87행):
```python
predict_vector = concatenate([dmf_vector, mlp_vector])
# dmf_vector: [batch, userlayers[-1]] = [batch, 64]
# mlp_vector: [batch, layers[-1]] = [batch, 64]
# predict_vector: [batch, 128]
```

**따라서**: Prediction layer input = `userlayers[-1] + layers[-1]` = 128

**원본 Step 1의 문제**:
- `dmf_layers[-1]` (64) + `dmf_layers[-1]` (64) = 128 ✅ (우연히 맞음)
- 하지만 `mlp_layers[-1]`를 사용해야 논리적으로 맞음

---

## 💡 현재 PyTorch 구현이 더 나은 이유

### 1. 직관적

```python
# DMF의 prediction 가중치 + MLP의 prediction 가중치 = CFNet의 prediction 가중치
new_weight = torch.cat([dmf_pred_weight, mlp_pred_weight], dim=1)
```

**의미**: CFNet은 DMF와 MLP를 concatenate하므로, 각각의 prediction 가중치도 concatenate하는 것이 자연스러움

---

### 2. 논리적 일관성

- Forward pass: `[dmf_vector, mlp_vector]` concatenate
- Pretrain 초기화: `[dmf_weights, mlp_weights]` concatenate
- **완벽한 대응**

---

### 3. 일반성

원본은 `userlayers[-1] == layers[-1]`일 때만 제대로 작동:
```python
# 만약 userlayers[-1]=64, layers[-1]=128이면?
# Step 1: [64, 1] + [64, 1](zeros) = [128, 1]  ← 잘못된 크기!
# Step 2: [128, 1][:128] + [128, 1] = [128 + 128, 1] = [256, 1]  ← 엉망!
```

현재 PyTorch 구현은 모든 경우에 작동:
```python
# userlayers[-1]=64, layers[-1]=128이면
new_weight = [1, 64] + [1, 128] = [1, 192]  ← 항상 올바른 크기
```

---

### 4. Forward pass와 100% 일치

가장 중요한 점: **Forward pass는 원본과 완전히 동일**

```python
# 원본 CFNet.py 87-91행
predict_vector = concatenate([dmf_vector, mlp_vector])
prediction = Dense(1, activation='sigmoid')(predict_vector)

# PyTorch cfnet_model.py 308-311행
predict_vector = torch.cat([dmf_vector, mlp_vector], dim=-1)
prediction = torch.sigmoid(self.prediction(predict_vector))
```

**결론**: Pretrain 초기화는 학습 **시작점**일 뿐, 모델 구조는 동일하므로 최종 성능에 큰 영향 없음

---

## 🔧 옵션 1: 원본 TensorFlow 로직 재현 (참고용)

필요 시 다음 코드로 원본 로직을 정확히 재현할 수 있습니다.

### 수정 방법

`cfnet/cfnet_model.py`의 `_load_pretrained_weights()` 메서드를 다음과 같이 수정:

```python
def _load_pretrained_weights(self, dmf_path, mlp_path):
    """
    사전 학습된 DMF와 MLP 모델의 가중치를 로드합니다.
    원본 TensorFlow 로직을 정확히 재현합니다.
    """
    # 파일 존재 확인
    if not os.path.exists(dmf_path):
        raise FileNotFoundError(f"DMF pretrain 파일을 찾을 수 없습니다: {dmf_path}")
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"MLP pretrain 파일을 찾을 수 없습니다: {mlp_path}")

    dmf_state = torch.load(dmf_path, map_location='cpu')
    mlp_state = torch.load(mlp_path, map_location='cpu')

    # ============================================================
    # DMF 가중치 로드 (원본 105-124 라인)
    # ============================================================

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

    # DMF prediction layer 가중치
    dmf_pred_weight = dmf_state['prediction.weight']  # [1, dmf_dim]
    dmf_pred_bias = dmf_state['prediction.bias']      # [1]

    # 원본 121-122행 로직: DMF 가중치 + 0으로 채우기
    dmf_dim = dmf_pred_weight.shape[1]  # userlayers[-1]
    zeros_part = torch.zeros(1, dmf_dim)  # [1, dmf_dim]

    # Step 1 가중치: [1, dmf_dim + dmf_dim]
    step1_weight = torch.cat([dmf_pred_weight, zeros_part], dim=1)
    step1_bias = dmf_pred_bias

    # ============================================================
    # MLP 가중치 로드 (원본 126-145 라인)
    # ============================================================

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

    # 원본 141-144행 로직: DMF 일부 + MLP 전체
    mlp_dim = mlp_pred_weight.shape[1]  # layers[-1]

    # Step 1 가중치의 앞 mlp_dim개만 사용
    dmf_part = step1_weight[:, :mlp_dim]  # [1, mlp_dim]

    # 최종 가중치: [1, mlp_dim + mlp_dim]
    new_weight = torch.cat([dmf_part, mlp_pred_weight], dim=1)
    new_bias = step1_bias + mlp_pred_bias

    # 0.5 scaling (원본 144행)
    self.prediction.weight.data.copy_(0.5 * new_weight)
    self.prediction.bias.data.copy_(0.5 * new_bias)
```

---

### 원본 재현 시 주의사항

#### 1. userlayers[-1] != layers[-1]인 경우

원본 로직은 다음 조건을 **가정**:
```python
userlayers[-1] == itemlayers[-1] == layers[-1]
```

이 조건이 깨지면 원본 로직이 제대로 작동하지 않음:

**예시**: `userlayers=[512, 64]`, `itemlayers=[1024, 64]`, `layers=[512, 256, 128, 128]`

```python
# DMF output: 64
# MLP output: 128
# CFNet prediction input: 64 + 128 = 192

# 원본 Step 1: [64, 1] + [64(zeros), 1] = [128, 1]  ← 잘못됨! (192여야 함)
# 원본 Step 2: [128, 1][:128] + [128, 1] = [256, 1]  ← 완전히 잘못됨!
```

#### 2. TensorFlow와 PyTorch의 shape 차이

- TensorFlow: `[output_dim, 1]`
- PyTorch: `[1, output_dim]` (transpose)

원본 재현 시 주의 필요

---

## 🎯 권장 사항

### ✅ 현재 구현 (옵션 2) 유지

**이유**:
1. **Forward pass 동일**: 모델 아키텍처는 원본과 100% 일치
2. **더 직관적**: DMF + MLP를 단순 concatenate
3. **논리적 일관성**: Forward와 pretrain 초기화 방식이 대응
4. **일반성**: 모든 layer 크기 조합에 대해 올바르게 작동
5. **Pretrain은 초기화**: 학습 후에는 수렴하므로 최종 성능에 큰 영향 없음

### ⚠️ 원본 재현 (옵션 1)은 다음 경우에만 고려

- 원본 TensorFlow 코드와 **완전히 동일한 실험 재현**이 필요한 경우
- 논문 저자의 공식 구현을 **그대로** 따라야 하는 경우
- 하지만 원본 로직 자체가 비일관적이므로 권장하지 않음

---

## 📚 참고: 원본 TensorFlow 코드 위치

- **파일**: `/Users/yeonghyeonchoe/dev/python/DeepCF-master/CFNet.py`
- **DMF pretrain 로드**: 105-124행 (`load_pretrain_model1`)
- **MLP pretrain 로드**: 126-145행 (`load_pretrain_model2`)
- **Forward pass**: 49-96행 (`get_model`)

---

## 🔬 실험 검증 (선택 사항)

두 방식의 성능 차이를 확인하려면:

1. 현재 구현 (옵션 2)로 학습 → 성능 A 기록
2. 옵션 1로 수정 후 학습 → 성능 B 기록
3. A와 B 비교:
   - 차이가 미미하면 (< 0.01): 옵션 2 유지
   - 차이가 크면 (> 0.05): 원본 로직 검토 및 논문 확인

**예상**: 차이 거의 없음 (pretrain은 초기화일 뿐)

---

**작성일**: 2025-10-16
**버전**: 1.0
**상태**: 현재 구현 (옵션 2) 유지 권장
