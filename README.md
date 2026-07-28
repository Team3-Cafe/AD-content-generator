# AD Content Generator

상품 이미지와 매장 정보를 입력받아 광고 이미지, 광고 문구, 문구 레이아웃까지 자동으로 생성하는 소상공인용 광고 콘텐츠 제작 파이프라인입니다.

## Tech Stack

| 구분 | 기술 |
|---|---|
| Language | Python 3.11~3.12 |
| UI | Streamlit |
| Vision / Image | Pillow, OpenCV, rembg, ONNX Runtime |
| Generative AI | OpenAI API, Diffusers, Stable Diffusion, ControlNet |
| Base Model | `digiplay/majicMIX_realistic_v7` |
| ML Runtime | PyTorch, Transformers, Accelerate, Safetensors |
| Evaluation | CLIP, DINO, LAION Aesthetic Predictor, HPS v2 |
| Infrastructure | GCP VM, NVIDIA L4 GPU |

## 1. 프로젝트 개요

기존 광고 콘텐츠 제작 과정에서는 상품 누끼 제거, 배경 제작, 광고 문구 작성, 문구 배치 작업을 각각 수행해야 합니다.

본 프로젝트는 생성형 AI와 이미지 처리 기술을 활용하여 다음 과정을 하나의 파이프라인으로 자동화합니다.

- 상품 이미지 전처리 및 배경 제거
- 상품과 매장 정보 기반 광고 장면 설계
- 상품의 구조와 깊이를 반영한 배경 생성
- 상품 원본 복원 및 경계 보정
- 광고 문구 생성
- 이미지 분석 기반 문구 레이아웃 생성
- 최종 결과 정량 평가

## 2. 주요 기능

- 상품 이미지 자동 전처리 및 배경 제거
- GPT Vision 기반 상품 이미지 분석
- 상품 및 매장 정보 기반 광고 배경 프롬프트 생성
- 상품 강조도와 브랜드 분위기 조절
- 입력 이미지 비율을 반영한 출력 해상도 자동 설정
- Canny / Depth ControlNet 기반 조건부 이미지 생성 및 원근 제어
- Inpainting 기반 상품 주변 배경 생성
- 상품 내부 원본 복원 및 경계 블렌딩
- 제목, 부제목, 가격, CTA 광고 문구 생성
- 상품·매장 정보 기반 광고 카피 생성
- 2단계 VLM 검토 기반 문구 레이아웃 생성
- GPU 추론 큐와 생성 모델 재사용
- 생성 이미지 정량 평가

## 3. 전체 파이프라인

<img width="1693" height="929" alt="AD Content Generator pipeline" src="https://github.com/user-attachments/assets/d99e0b8f-fc02-4072-b46e-89fe5f8dd759" />

파이프라인은 크게 이미지 생성 단계와 카피·레이아웃 단계로 구분됩니다.

1. 상품 이미지에서 누끼와 마스크를 생성합니다.
2. GPT Vision이 상품과 매장 정보를 분석해 장면 프롬프트와 레이아웃을 설계합니다.
3. Canny, Depth, Inpainting 조건을 사용해 상품 주변 장면을 생성합니다.
4. 상품 내부와 경계를 복원해 원본 정체성을 보존합니다.
5. 광고 문구를 생성하고 VLM이 최종 이미지에 문구를 배치합니다.

## 4. 이미지 생성 모델 구성

| 구성 요소 | 모델 및 설정 |
|---|---|
| Base Model | `digiplay/majicMIX_realistic_v7` |
| Canny ControlNet | `lllyasviel/control_v11p_sd15_canny` |
| Depth ControlNet | `lllyasviel/control_v11f1p_sd15_depth` |
| Depth Estimator | `Intel/dpt-hybrid-midas` |
| Scheduler | Euler Ancestral |
| Inference Steps | 35 |
| Guidance Scale | 6.5 |
| Inpainting Strength | 0.80 |
| Canny Weight | 0.30 |
| Depth Weight | 0.30 |
| 기본 해상도 | Short side 512px, Long side 최대 1024px |

### 조건별 역할

- **Canny ControlNet**: 상품의 주요 윤곽과 구조를 유지합니다.
- **Depth ControlNet**: 상품과 지지면 사이의 깊이, 원근감, 카메라 시점을 반영합니다.
- **Inpainting**: 상품 영역을 보호하면서 상품 주변의 배경을 생성합니다.
- **Core Refinement**: 생성 과정에서 변형된 상품 내부를 원본 기반으로 복원합니다.
- **Identity Restoration**: 상품의 경계와 색감을 보정하여 배경과 자연스럽게 연결합니다.

## 5. 프로젝트 구조

```text
AD-content-generator/
├── adcg/
│   ├── preprocessing/     # 누끼 제거, Alpha Mask 생성, 이미지 검증
│   ├── prompting/         # 상품 분석, 장면 프롬프트, 광고 카피
│   ├── generation/        # Canvas, Canny, Depth, Inpainting 생성
│   ├── refinement/        # 상품 내부 복원 및 경계 보정
│   ├── prompt_layout/     # VLM 기반 문구 레이아웃 및 렌더링
│   ├── image_utils/       # 마스크, 블렌딩, 레이아웃 공통 함수
│   ├── eval/              # 이미지 정량 평가
│   ├── runtime/           # GPU 추론 큐 및 모델 재사용
│   ├── pipelines/         # 단계별 파이프라인 실행
│   ├── config.py          # CLI 입력 및 환경 설정
│   └── pipeline.py        # 전체 파이프라인 연결
├── tests/                 # 단위 및 파이프라인 테스트
├── docs/                  # 실험 기록 문서
├── run_pipeline.py        # 전체 파이프라인 CLI
├── app.py                 # Streamlit 서비스 UI
├── requirements.txt
├── .env.example
└── README.md
```

## 6. 설치 및 환경 설정

### 실행 환경

- Python 3.11~3.12
- CUDA 지원 GPU 권장
- GCP L4 GPU 환경에서 실험
- OpenAI API Key 필요

### 저장소 설치

```bash
git clone https://github.com/Team3-Cafe/AD-content-generator.git
cd AD-content-generator

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Windows PowerShell에서는 다음 명령으로 가상환경을 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

CUDA 서버에서는 환경의 CUDA 버전에 맞는 `torch`, `torchvision`을 먼저 설치하는 것을 권장합니다.

`rembg`를 GPU에서 실행하려면 `onnxruntime` 대신 `onnxruntime-gpu`를 설치해야 합니다.

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

현재 `requirements.txt`에는 웹 UI 의존성이 포함되어 있지 않습니다. `app.py`를 실행하려면 Streamlit을 추가로 설치합니다.

```bash
pip install "streamlit>=1.40,<2"
```

### 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
OPENAI_API_KEY=your_openai_api_key
GCP_PROJECT_ID=
GCP_REGION=
```

`.env`와 실제 API Key는 GitHub에 커밋하지 않습니다.

## 7. 실행 방법 및 산출물

### Streamlit 웹 애플리케이션

`app.py`는 상품 이미지 업로드, 상품·매장 정보 입력, 생성 옵션 설정, 광고 이미지 생성, 카피 및 레이아웃 생성까지 제공하는 서비스 진입점입니다.

```bash
streamlit run app.py
```

웹 UI의 처리 흐름은 다음과 같습니다.

1. 상품 이미지와 상품·매장 정보를 입력합니다.
2. 화면 비율, 레이아웃 모드, 상품 강조도, 상품 크기, 브랜드 강조도를 설정합니다.
3. 이미지 생성 요청을 제출하고 모델 준비, GPU 대기, GPU 추론 상태를 확인합니다.
4. 광고 문구의 톤과 길이를 선택합니다.
5. VLM 문구 레이아웃과 최종 광고 이미지를 확인합니다.

`app.py`는 다음 방식으로 다중 사용자 요청을 처리합니다.

- `@st.cache_resource`로 Diffusion 파이프라인과 추론 큐를 한 번만 생성해 재사용합니다.
- 사용자 세션별 UUID 기반 입출력 디렉터리로 산출물 충돌을 방지합니다.
- CPU 전처리와 프롬프트 생성을 먼저 수행한 뒤 GPU 작업을 큐에 등록합니다.
- GPU 이미지 생성 요청은 공유 FIFO 큐에서 순차 처리해 VRAM 충돌을 방지합니다.

### 전체 파이프라인 CLI

```bash
python run_pipeline.py \
  --image inputs/product.png \
  --info inputs/product_info.json \
  --output-dir outputs/pipeline/demo \
  --gpt-model gpt-5.4-nano \
  --product-focus 1.0 \
  --brand-focus 0.5 \
  --layout-mode layout \
  --copy-count 1 \
  --seed 42 \
  --cpu-offload
```

### 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--image` | 입력 상품 이미지 경로 |
| `--info` | 상품 및 매장 정보 JSON |
| `--output-dir` | 결과 저장 경로 |
| `--product-focus` | 상품 강조 정도, `0.0~1.0` |
| `--brand-focus` | 일상적 환경과 프리미엄 브랜드 배경의 조절값 |
| `--product-scale` | 전체 화면에서 상품이 차지하는 크기 |
| `--width`, `--height` | 출력 이미지 크기 |
| `--layout-mode` | `layout` 또는 `preserve` |
| `--copy-count` | 생성할 광고 문구 개수 |
| `--evaluate` | 정량 평가 실행 |
| `--eval-metrics` | 실행할 평가 지표 선택 |
| `--cpu-offload` | GPU 메모리 절약을 위한 CPU Offload |

`--info`를 생략하면 `--product-name`, `--store-name` 등의 CLI 입력으로 상품 정보 JSON을 생성할 수 있습니다. 이 경우 `--product-name`과 `--store-name`은 필수입니다.

### 주요 산출물

```text
outputs/pipeline/demo/
├── 00_config/
│   └── product_info.json
├── 01_preprocessed/
│   ├── product_cutout_trimmed.png
│   └── product_mask.png
├── 02_prompt/
│   ├── ad_prompt.json
│   └── ad_copy.json
├── 03_generated/
│   ├── condition_canvas.png
│   ├── product_canny_control.png
│   ├── product_depth_control.png
│   └── generated_with_cutout_condition.png
├── 05_final/
│   └── final_identity_restored.png
├── 06_eval/
│   └── eval_results.json
└── 07_prompt_layout/
    ├── layout.json
    ├── vlm_1_ad.png
    └── final_ad.png
```

- `ad_prompt.json`: 상품 분석, 배경 프롬프트, 레이아웃 정보
- `ad_copy.json`: 제목, 부제목, 가격, CTA 문구
- `final_identity_restored.png`: 상품 복원까지 완료된 광고 배경
- `layout.json`: 최종 문구 위치, 크기, 색상, 스타일
- `vlm_1_ad.png`: 1차 VLM 레이아웃 결과
- `final_ad.png`: 2차 VLM 검토를 거친 최종 광고 이미지
- `eval_results.json`: 선택한 정량 평가 지표 결과

## 8. 모델 재사용 및 GPU 추론 큐

`adcg.runtime.InferenceQueue`는 생성 모델을 요청마다 다시 로드하지 않고 한 번 로드한 뒤 여러 요청에서 재사용합니다.

### 처리 방식

1. 상품 전처리와 GPT 프롬프트 생성을 CPU에서 병렬로 수행합니다.
2. 준비가 끝난 작업은 GPU 대기 큐에 등록됩니다.
3. GPU 작업은 FIFO 방식으로 한 번에 하나씩 실행됩니다.
4. MajicMIX와 ControlNet 모델은 GPU 메모리에 유지됩니다.
5. 작업이 끝나면 불필요한 PIL 이미지와 중간 객체를 해제합니다.

### 작업 상태

| 상태 | 설명 |
|---|---|
| `preparing` | 전처리 및 프롬프트 준비 중 |
| `waiting_gpu` | GPU 실행 대기 중 |
| `running_gpu` | 이미지 생성 진행 중 |
| `completed` | 생성 완료 |
| `failed` | 실행 실패 |
| `cancelled` | 작업 취소 |

이 구조는 모델 중복 로드와 GPU 메모리 충돌을 줄이고, 여러 사용자가 요청하는 환경에서도 안정적으로 이미지 생성 작업을 처리하도록 설계되었습니다.
