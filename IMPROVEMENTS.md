# 🚀 Baseline 모델 개선 전략 및 구현

## 📊 개선 요약

### **Phase 1: 데이터 전략 (가장 중요!)**
| 항목 | Baseline | 개선 | 이유 |
|------|----------|------|------|
| 학습 샘플 수 | 500,000 | **1,000,000** | 더 많은 패턴 학습 |
| 학습 자릿수 | 1~3자리 | **1~5자리** | OOD 간극 최소화 (규정 최대치) |
| 복잡도 (depth) | 3 | **4** | 복잡한 수식 학습 |
| 검증 샘플 수 | 128 | **256** | 더 신뢰성 있는 평가 |

**핵심**: 학습 데이터를 1~3자리 → 1~5자리로 확대하면, 평가 시 6자리+ OOD 일반화 성능이 **대폭 향상**됩니다!

---

### **Phase 2: 아키텍처 개선**

#### **Transformer vs GRU**
```
GRU (Baseline):
  입력 전체 → 단일 벡터 → 출력
  ❌ 병목: 긴 시퀀스 정보 손실
  ❌ 위치 정보 부족

Transformer (개선):
  입력 전체 ← Self-Attention → 출력
  ✅ 모든 위치 참조 가능
  ✅ Positional Encoding (위치 정보 명시적)
  ✅ 병렬 학습 가능
```

| 항목 | Baseline (GRU) | 개선 (Transformer) |
|------|----------------|-------------------|
| 아키텍처 | GRU Encoder-Decoder | Transformer Encoder-Decoder |
| d_model | 256 | **512** |
| Layers | 1 | **4 encoder + 4 decoder** |
| Attention | ❌ | ✅ Multi-head (8 heads) |
| Positional Encoding | ❌ | ✅ Learnable |
| 파라미터 수 | ~800K | **~25M** |

---

### **Phase 3: 학습 최적화**

| 항목 | Baseline | 개선 | 효과 |
|------|----------|------|------|
| Learning Rate | 2e-3 | **3e-4** | 안정적 학습 |
| LR Schedule | 고정 | **Warmup + Cosine** | 빠른 수렴 |
| Label Smoothing | 0.0 | **0.1** | Overconfidence 방지 |
| Batch Size | 128 | **64** | Transformer 메모리 최적화 |
| Epochs | 4~50 | **10** | Transformer는 빠르게 수렴 |
| Max Output Length | 32 | **50** | 긴 숫자 출력 가능 |

---

## 🎯 예상 성능 개선

### **현재 Baseline 성능**
- EM (Exact Match): **36.7%**
- TES (Token Edit Similarity): **59.2%**

### **개선 목표**
```
단계별 예상 성능:

1. 데이터만 개선 (1~5자리):
   EM: 36.7% → 55~60% (+18~23%)
   
2. Transformer 추가:
   EM: 55~60% → 70~75% (+15%)
   
3. 학습 최적화 추가:
   EM: 70~75% → 80~85% (+10%)

최종 목표: EM 80%+ (현재 대비 2.2배 향상)
```

---

## 📝 주요 변경사항 상세

### **1. config.py**

```python
# ModelConfig 확장
@dataclass
class ModelConfig:
    d_model: int = 512              # 256 → 512
    nhead: int = 8                  # NEW
    num_encoder_layers: int = 4     # NEW
    num_decoder_layers: int = 4     # NEW
    dim_feedforward: int = 2048     # NEW
    dropout: float = 0.1            # NEW
    use_transformer: bool = True    # NEW

# TrainConfig 개선
@dataclass
class TrainConfig:
    lr: float = 3e-4               # 2e-3 → 3e-4
    warmup_steps: int = 1000       # NEW
    valid_every: int = 200         # 50 → 200
    max_gen_len: int = 50          # 32 → 50
    num_epochs: int = 10           # 4 → 10
    label_smoothing: float = 0.1   # NEW
    grad_clip: float = 1.0         # NEW
```

### **2. model.py - TransformerSeq2Seq 추가**

```python
class TransformerSeq2Seq(nn.Module):
    """Transformer 기반 Seq2Seq (성능 개선 버전)"""
    
    def __init__(self, in_vocab, out_vocab, **kwargs):
        # 임베딩
        self.embed_in = nn.Embedding(in_vocab, d_model)
        self.embed_out = nn.Embedding(out_vocab, d_model)
        
        # Positional Encoding (학습 가능)
        self.pos_encoder = nn.Embedding(512, d_model)
        self.pos_decoder = nn.Embedding(512, d_model)
        
        # Transformer
        self.encoder = nn.TransformerEncoder(...)
        self.decoder = nn.TransformerDecoder(...)
        
        # Output
        self.out_proj = nn.Linear(d_model, out_vocab)
```

**핵심 개선점**:
- ✅ Multi-head Self-Attention (전체 시퀀스 참조)
- ✅ Positional Encoding (위치 정보 명시적 학습)
- ✅ Feed-Forward Networks (비선형 변환 강화)
- ✅ Layer Normalization (안정적 학습)
- ✅ Residual Connections (Gradient flow 개선)

### **3. train.py - 학습 루프 개선**

```python
# 데이터 개선
train_dataset = ArithmeticDataset(
    num_samples=1_000_000,  # 500K → 1M
    max_depth=4,            # 3 → 4
    num_digits=(1, 5),      # (1,3) → (1,5) ⭐ 핵심!
)

# Learning Rate Scheduler
def get_lr(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps  # Linear warmup
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return base_lr * 0.5 * (1 + cos(progress * π))  # Cosine annealing

# Label Smoothing
loss_fn = nn.CrossEntropyLoss(
    ignore_index=pad_id,
    label_smoothing=0.1,  # NEW
)
```

---

## 🔬 기술적 세부사항

### **Transformer의 Attention 메커니즘**

```
입력: "12+34"

Step 1: Self-Attention (Encoder)
  '1' ← 모든 토큰 참조 → ['1', '2', '+', '3', '4']
  '2' ← 모든 토큰 참조 → ['1', '2', '+', '3', '4']
  '+' ← 모든 토큰 참조 → ['1', '2', '+', '3', '4']
  ...
  
결과: 각 위치가 필요한 정보를 선택적으로 가져옴

Step 2: Cross-Attention (Decoder)
  출력 '4' ← 입력 전체 참조 → ['1', '2', '+', '3', '4']
  출력 '6' ← 입력 전체 참조 → ['1', '2', '+', '3', '4']
  
결과: 출력이 입력의 어떤 부분과 관련되는지 학습
```

### **Positional Encoding의 역할**

```
입력: "123" vs "321"

Positional Encoding 없으면:
  embed('1') + embed('2') + embed('3') = 같은 표현 (순서 무시)
  
Positional Encoding 있으면:
  embed('1') + pos(0) + embed('2') + pos(1) + embed('3') + pos(2)
  → 위치 정보 명시적으로 학습
  → "첫 번째 자릿수", "백의 자리" 등 개념 학습 가능
```

---

## 📈 성능 향상의 핵심 메커니즘

### **1. OOD 간극 감소**
```
Baseline:
  학습: 1~3자리 (예: 1, 12, 123)
  평가: 6자리+ (예: 123456)
  간극: 3자리 → 6자리 (2배 도약) ❌

개선:
  학습: 1~5자리 (예: 1, 12, 123, 1234, 12345)
  평가: 6자리+ (예: 123456)
  간극: 5자리 → 6자리 (1.2배 도약) ✅
  
결과: 모델이 5자리 패턴을 학습 → 6자리로 일반화 훨씬 쉬움!
```

### **2. Attention의 효과**
```
문제: "2+3*4" = 14 (연산자 우선순위)

GRU:
  '2' → h1 → h2 → h3 → h4 → h5 (순차적 압축)
  h5 하나로 모든 정보 표현 ❌
  
Transformer:
  출력 생성 시:
  - Attention('*') → '+' 보다 우선
  - '3'과 '4'를 먼저 결합
  - 결과를 '2'와 합산
  ✅ 연산자 우선순위 명시적 학습!
```

### **3. Positional Encoding 효과**
```
문제: 자릿수 올림/내림

"999+1" = 1000

Positional Encoding 있으면:
  pos(0): 일의 자리 → 올림 발생 학습
  pos(1): 십의 자리 → 올림 전파 학습
  pos(2): 백의 자리 → 올림 전파 학습
  pos(3): 천의 자리 → 새 자릿수 생성 학습
  
✅ 자릿수별 독립적 처리 가능!
```

---

## 🎯 대회 평가 지표별 전략

### **1. Calculation Accuracy (35%)**
- ✅ 1~5자리 학습으로 기본 연산 정확도 향상
- ✅ Transformer로 복잡한 수식 처리

### **2. Law Preservation (20%)**
- ✅ Attention으로 연산자 우선순위 학습
- ✅ 더 많은 샘플로 교환/결합 법칙 학습

### **3. Expression Consistency (30%)**
- ✅ Positional Encoding으로 구조적 이해
- ✅ Self-Attention으로 동치 표현 인식

### **4. Relational Consistency (15%)**
- ✅ 자릿수 정보 명시적 학습
- ✅ 출력 간 관계 유지 능력 향상

---

## 🚀 실행 방법

```bash
# 1. 개선된 모델 학습
python3 train.py

# 예상 학습 시간:
# - CPU: ~8시간
# - GPU (CUDA): ~1시간

# 2. 로컬 테스트
python3 local_test.py .

# 3. 체크포인트 확인
ls -lh best_model.pt
```

---

## 📊 예상 결과

```
Epoch 1: EM ~50% (기본 패턴 학습)
Epoch 3: EM ~65% (복잡한 패턴 학습)
Epoch 5: EM ~75% (일반화 시작)
Epoch 8: EM ~80%+ (목표 달성)

최종 체크포인트:
- 파일 크기: ~100MB (Baseline 3MB → 33배)
- EM: 80~85%
- TES: 90~95%
```

---

## 💡 추가 개선 아이디어 (시간 있을 때)

1. **Beam Search**: Greedy → Beam Search (k=5)
2. **Data Augmentation**: 괄호 추가/제거, 연산자 교환
3. **Curriculum Learning**: 쉬운 문제 → 어려운 문제
4. **Auxiliary Loss**: 중간 계산 단계 예측
5. **Larger Model**: d_model 512 → 768
6. **More Data**: 1M → 2M 샘플

---

## 🎉 결론

이번 개선으로:
- ✅ **데이터**: 1~5자리까지 학습 (OOD 간극 최소화)
- ✅ **아키텍처**: Transformer (정보 병목 해결)
- ✅ **학습**: Warmup + Label Smoothing (안정적 최적화)

**예상 성능**: EM 36.7% → **80%+** (2.2배 향상)

**대회 목표**: 수학적 일반화 능력 검증 → **달성 가능!**
