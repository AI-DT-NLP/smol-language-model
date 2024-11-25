### Simple Roadmap **Implementation Steps / 구현 단계**

#### 1. **Base Model Development / 기본 모델 개발**
- [ ] Transformer Decoder
  - Multi-Head Attention
  - Feed Forward Layer
  - Residual Connections & Layer Normalization

---

#### 2. **Dataset and Preprocessing / 데이터셋 및 전처리**

##### **Dataset Sources / 데이터셋 소스**
1. **General Text Data / 일반 텍스트 데이터**:
   - Webpages: CommonCrawl, OpenWebText, C4
   - Books: BookCorpus, Gutenberg
   - Wikipedia: Multi-language articles
(해당 데이터셋은 데이터 셋의 품질 보증X)

2. **Specialized Data / 특화 데이터**:
   - Scientific: arXiv, Semantic Scholar (not selected)
   - Code: [Tiny Code](https://huggingface.co/datasets/nampdn-ai/tiny-codes) 

##### **Preprocessing Techniques / 전처리 기술**
1. **Data Filtering / 데이터 필터링**:
   - Language, keyword, perplexity-based filtering
   - Classifier-based filtering for quality control
2. **Deduplication / 중복 제거**:
   - Applied at sentence, document, dataset levels
3. **Privacy Reduction / 개인 정보 보호**:
   - Remove PII using rule-based techniques
4. **Tokenization / 토크나이제이션**:
   - SentencePiece or Byte-Pair Encoding (BPE)

---

#### 3. **Pre-Training / 사전 학습**

1. **Optimization / 최적화**:
   - Optimizer: AdamW
   - Scheduler: Warm-up + Cosine Decay
2. **Precision / 연산 정밀도**:
   - Mixed Precision Training (FP16)
3. **Gradient Accumulation / 그래디언트 누적**:
   - Simulate larger batch sizes by accumulating gradients.

---

#### 4. **Post-Training Enhancements / 사후 학습 개선**

1. **Supervised Fine-Tuning (SFT)**
   - Align the model with domain-specific datasets for task-specific improvements.  
    - CodeExercises Dataset
   - 도메인 특화 데이터셋으로 모델을 튜닝하여 작업 효율 향상.

2. **Reinforcement Learning with Human Feedback (RLHF)**
   - Incorporate human annotations to refine output quality.  
   - 사용자 피드백을 활용하여 출력 품질을 개선.

---

#### 5. **Evaluation / 평가**

1. **General Benchmarks / 일반 벤치마크**:
   - Perplexity, BLEU, ROUGE
2. **Emergent Abilities / 특이 능력**:
   - In-context learning, instruction following

---

### **References / 참고자료**

1. **Papers / 논문**
   - ["Textbook Is All You Need"](https://arxiv.org/abs/2306.11644)
   - ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
   - ["RLHF - Training Language Models to Follow Instructions"](https://arxiv.org/abs/2203.02155)
   - ["Direct Preference Optimization"](https://arxiv.org/abs/2305.18290)

2. **Guides / 가이드**
   - [HuggingFace `transformers`](https://huggingface.co/transformers/)
   - [“The Annotated Transformer”](http://nlp.seas.harvard.edu/annotated-transformer/)
