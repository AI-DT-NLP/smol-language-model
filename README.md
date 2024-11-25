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

##### TODO

- [ ] pre-trained data와 fine-tuning data 선택
    - [Tiny Code](https://huggingface.co/datasets/nampdn-ai/tiny-codes)

##### Record

- [hypothesis]: How to develop smol language model? it can build on 16VRAM GPU?
- [experiment]: Develop a baseline model based on GPT-1 that can train on 16GB VRAM. but dataset is specialized and high quality. 
- [conclusion]: 
 

---

##### directory structure


```
slm
├── README.md
├── datasets
│   └── preprocessed
├── get_dataset.py
├── model
│   ├── baseline.py
│   ├── checkpoints
│   └── model_explain.md
├── preprocessing
│   └── baseline.py
├── run.py
├── select framework.md
├── simple roadmap.md
├── slm training parameter with vram 16gb.md
└── train
    └── baseline.py
```