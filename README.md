# LLM_formal_convertor
한국말의 고유 특성을 고려하기 위해 문장을 존댓말로 변환하는 변환기(convertor) 구성 프로젝트

---

## Setup
- Python 3.7.0 ~ 3.7.9
- CUDA Version 11.0

1. Nvidia driver, CUDA toolkit 11.0, install Anaconda.

2. Install pytorch

        pip install transformers
	
## Training

When using Terminal, directly execute the code below after setting the path

	python train.py --kernel-type model_name --out-dim 4 --data-folder images/ --enet-type preconv_seven --n-epochs 200 --init-lr 4e-5 --batch-size 32 --k-fold 0 --image-size 256 --CUDA_VISIBLE_DEVICES 0
