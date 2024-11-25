## **Building a Lightweight LLM from Scratch**

### **Overview**  
This project aims to build a small language model (SLM) with up to 100M parameters under a VRAM 16GB environment. By simplifying the key technologies of large-scale language models, this project explores the practicality of lightweight models in NLP.

이 프로젝트는 VRAM 16GB 환경에서 최대 1억 개의 파라미터를 가지는 소형 언어 모델(SLM)을 구축하는 것을 목표로 합니다. 다음은 대규모 언어 모델의 핵심 기술을 간소화하여 보여줍니다.

---

### How to start

1. make new python enviroment(or conda) `conda create -n slm python=3.9 -y`
2. set your new enviroment and `pip install torch transformers datasets`
3. `python get_dataset.py` and `python run.py`. this is run all baseline(preprocessing->model->train)

---

### TODO

- [ ] pre-trained data와 fine-tuning data 선택
    - 

---

### Roadmap **Implementation Steps / 구현 단계**

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
   - __Code: [Tiny Code](https://huggingface.co/datasets/nampdn-ai/tiny-codes) (selected)__  

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
