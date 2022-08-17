# OSLO로 KoGPT 분산 학습하기

대규모 분산학습라이브러리인 OSLO의 TensorParallel을 사용하여 
모델을 쪼개서 학습하는 예제.

## 스펙
모델 : 카카오브레인 KoGPT (fp32)
데이터셋 : KorQuADv1


## OSLO 설치
### case1. 파일사용
`bash install.sh`

### case2. 수동설치
```bash
git clone https://github.com/tunib-ai/oslo.git
git checkout 85a4ff11816f8319a7344f1e596dd6b3e7592034

cd oslo
pip install --editable .
```

## 학습하기 (싱글노드, 4gpu)
```python -m torch.distributed.launch --nproc_per_node=4 finetune.py```

## 학습하기 (멀티노드, 8gpu=4gpu x 2)
```bash
# 1번노드
python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=4 --master_addr=${YOUR_NODE_ADDRESS} --master_port=${PORT} finetune.py

# 2번노드
python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=4 --master_addr=${YOUR_NODE_ADDRESS} --master_port=${PORT} finetune.py
```

### 참고
```bash
--nnodes : 전체 노드 개수
--node_rank : 노드의 우선순위, 0이 마스터노드
--nproc_per_node : 노드당 프로세스(gpu)개수
```
