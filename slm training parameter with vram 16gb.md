
### 16GB VRAM에서 Pre-train 가능한 모델 분석

---

## 1. 기본 계산 요소

1. **파라미터 크기 (VRAM 요구량)**:
   - 파라미터 크기(P): \( P \times 4 \) bytes (FP32 기준).
   - Mixed Precision (FP16) 적용 시 절반인 \( P \times 2 \) bytes.

2. **추가 메모리 요구량**:
   - **Optimizer States**: 일반적으로 모델 크기의 2~3배 추가 메모리 요구.
   - **Activation Memory**: 배치 크기와 시퀀스 길이에 비례.

3. **추가 기법으로 최적화**:
   - **Gradient Checkpointing**: Activation Memory를 줄일 수 있음.
   - **Deepspeed ZeRO Stage 2/3**: Optimizer States를 CPU로 분산 가능.
   - **FP16**: 메모리 절약의 핵심.

---

## 2. 모델별 메모리 요구량 및 분석

| 모델                  | 파라미터 수 | VRAM 요구량 (FP16) | 배치 크기 1 학습 가능성 | 배치 크기 > 1 학습 가능성 |
|-----------------------|-------------|--------------------|--------------------------|---------------------------|
| **CodeGen-Mono-350M** | 350M        | ~0.7GB            | 가능                     | 가능                      |
| **phi-1-small 350M**  | 350M        | ~0.7GB            | 가능                     | 가능                      |
| **phi-1-base 1.3B**   | 1.3B        | ~2.6GB            | 가능                     | 가능                      |
| **Replit 2.7B**       | 2.7B        | ~5.4GB            | 가능                     | 제한적 (최적화 필요)      |
| **StarCoder 15.5B**   | 15.5B       | ~31GB             | 불가능                  | 불가능                   |
| **CodeGen-Mono-16.1B**| 16.1B       | ~32GB             | 불가능                  | 불가능                   |


---

### **현실적인 모델 파라미터 설정**

#### 1. **모델 크기**

- 목표: **100M (약 1억 개의 파라미터)**
    - Transformer Decoder 아키텍처의 경우, 주요 구성 요소:
        - Embedding Layer
        - Multi-Head Attention
        - Feed Forward Network
    - 파라미터 분배 (예시):
        - **Vocabulary Size**: 32,000 (SentencePiece 또는 BPE 기반)
        - **Hidden Size**: 512 (임베딩 크기)
        - **Number of Layers**: 12
        - **Number of Attention Heads**: 8
        
        > 계산식:
        
        - Embedding Layer: Vocab Size×Hidden Size=32K×512\text{Vocab Size} \times \text{Hidden Size} = 32K \times 512Vocab Size×Hidden Size=32K×512 ≈ **16M**
        - Attention + Feed Forward: 약 **7M/Layer** × 12 Layers ≈ **84M**
        - 전체 모델: 약 **100M 파라미터**

---

#### 2. **Batch Size**

- VRAM 16GB 기준으로 적합한 **Global Batch Size**:
    - Mixed Precision Training (FP16) 사용 시 약 **4~8 개 샘플/배치**.
    - Gradient Accumulation을 활용해 효과적인 Batch Size를 증가 가능:
        - e.g., Accumulation Step 8 → Effective Batch Size 64.

---

### **현실적인 데이터셋 크기**

#### 1. **훈련 데이터 크기**

- 파라미터 크기에 따른 데이터 크기 계산:
    - 일반적으로 훈련 데이터 크기≈20×모델 파라미터 수\text{훈련 데이터 크기} \approx 20 \times \text{모델 파라미터 수}훈련 데이터 크기≈20×모델 파라미터 수.
    - 100M 모델 기준: 약 **20억 토큰** 필요.

#### 2. **훈련 샘플 크기**

- 평균 50토큰/샘플로 가정할 때:
    - 20억 토큰÷50 토큰/샘플≈4천만 샘플.

#### 3. **데이터셋 탐색**

- 훈련 데이터 소스:
    - **
- 데이터 크기 선택:
    
    - VRAM 16GB와 훈련 시간 제한을 고려하여:
        - 약 **5~10GB (10억 토큰)**로 데이터 축소.