import streamlit as st
import os
import json
import time
import traceback
from pathlib import Path
from uuid import uuid4

# 팀 프로젝트 내부 이미지 파이프라인과 FIFO 큐
from adcg.generation import GENERATION_DEFAULTS, load_generation_pipeline
from adcg.runtime import InferenceQueue

# 1. 페이지 설정
st.set_page_config(page_title="AI 로 만드는 우리 가게 광고", layout="wide")


@st.cache_resource(show_spinner=False)
def get_inference_queue():
    """Keep one diffusion model and one FIFO worker per app process."""

    def load_pipe():
        return load_generation_pipeline(
            base_model=GENERATION_DEFAULTS["base_model"],
            controlnet_model=GENERATION_DEFAULTS[
                "controlnet_model"
            ],
            cpu_offload=False,
        )

    return InferenceQueue(pipe_factory=load_pipe)


# 첫 실행에서만 큐와 모델을 만들고, Streamlit rerun 중에는 재사용한다.
inference_queue = get_inference_queue()

# 2. PPT 29-34 톤앤매너 프리미엄 CSS 적용
st.html(
"""
<style>
/* 전체 앱 및 사이드바 배경 (웜 화이트 & 소프트 베이지) */
.stApp {
    background:
        radial-gradient(circle at 92% 2%, rgba(211, 84, 0, 0.10), transparent 28rem),
        linear-gradient(145deg, #FDFBF7 0%, #F8F2EA 100%);
}
.block-container {
    max-width: 1600px;
    padding-top: 1.25rem;
    padding-bottom: 1.5rem;
}
[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}
.block-container h1 {
    font-size: clamp(1.8rem, 2.5vw, 2.7rem) !important;
    line-height: 1.2 !important;
    margin-bottom: 0.25rem !important;
}
.block-container h3 {
    font-size: 1.15rem !important;
    margin-bottom: 0.15rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    padding: 0.75rem !important;
    background: #FFFDFC !important;
    border: 1px dashed #CDB9A5 !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F5EDE4 0%, #EFE2D5 100%) !important;
    border-right: 1px solid #E6DDD0;
}

.step-hero {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid rgba(196, 169, 143, 0.50);
    border-radius: 18px;
    box-shadow: 0 14px 40px rgba(89, 63, 42, 0.08);
    padding: 1rem 1.25rem 1.05rem;
    margin: 0 0 0.7rem;
    backdrop-filter: blur(10px);
}
.step-track {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
}
.step-chip {
    border: 1px solid #DDCFC2;
    border-radius: 999px;
    color: #76685A !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.28rem 0.62rem;
}
.step-chip.done {
    background: #F3E3D4;
    border-color: #E5C8AD;
    color: #A24616 !important;
}
.step-chip.active {
    background: #D35400;
    border-color: #D35400;
    color: #FFFFFF !important;
    box-shadow: 0 5px 14px rgba(211, 84, 0, 0.24);
}
.step-hero h1 {
    color: #26221F !important;
    font-size: clamp(1.55rem, 2.2vw, 2.35rem) !important;
    line-height: 1.2 !important;
    margin: 0 0 0.3rem !important;
}
.step-hero p {
    color: #71665D !important;
    font-size: 0.92rem;
    margin: 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.78);
    border-color: #E2D5C8 !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(89, 63, 42, 0.06);
}
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid #E2D5C8 !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(89, 63, 42, 0.06);
    padding: 1rem !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div {
    background-color: #FFFDFC !important;
    border-color: #D9CABC !important;
    border-radius: 9px !important;
}

/* 타이틀 및 텍스트 가독성 고정 (딥 차콜) */
h1, h2, h3, h4, h5, h6, p, span, label {
    color: #2B2B2B !important;
    font-family: 'Noto Sans KR', sans-serif;
}

/* STEP 포인트 라벨 (테라코타 오렌지) */
h4[id^="step"] {
    color: #D35400 !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}

/* 입력 필드 디자인 (화이트 박스 + 세련된 테두리) */
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1C7BD !important;
    border-radius: 8px !important;
}

/* 입력 필드 포커스 시 오렌지 테두리 */
[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
    border-color: #D35400 !important;
}

/* 입력창 내부 글자색 완벽 고정 (가독성 확보) */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #2B2B2B !important;
    -webkit-text-fill-color: #2B2B2B !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] svg {
    fill: #6E655B !important;
}

/* 상단 헤더 투명화 */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* 실행 버튼 디자인 (포인트 테라코타 오렌지) */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #D35400 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: bold !important;
    box-shadow: 0 2px 4px rgba(211, 84, 0, 0.2) !important;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #E67E22 !important;
}
div.stButton > button p,
div[data-testid="stFormSubmitButton"] > button p {
    color: #FFFFFF !important;
}

/* 파일 업로더 내 버튼 */
[data-testid="stFileUploader"] button {
    background-color: #2B2B2B !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
}

/* 안내 메시지 상자 */
div.stWarning, div.stInfo {
    background-color: #F4EBE1 !important;
    border: 1px solid #D1C7BD !important;
    border-radius: 8px !important;
}
div.stWarning *, div.stInfo * {
    color: #D35400 !important;
}

/* 역할이 분명한 접근성 중심 컬러 시스템 */
:root {
    --ui-ink: #24323B;
    --ui-muted: #68747A;
    --ui-primary: #C65D32;
    --ui-primary-hover: #A94725;
    --ui-secondary: #3F6B73;
    --ui-success: #3F765B;
    --ui-canvas: #F7F2EC;
    --ui-surface: #FFFDF9;
    --ui-border: #DED2C5;
}
.stApp {
    background:
        radial-gradient(circle at 92% 3%, rgba(198, 93, 50, 0.13), transparent 26rem),
        radial-gradient(circle at 8% 92%, rgba(63, 107, 115, 0.09), transparent 30rem),
        linear-gradient(145deg, #FBF8F4 0%, var(--ui-canvas) 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E8EFF0 0%, #DDE7E8 100%) !important;
    border-right: 1px solid #C9D7D9 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: var(--ui-ink) !important;
}
.step-hero {
    background: linear-gradient(120deg, rgba(255, 253, 249, 0.96), rgba(242, 232, 221, 0.90));
    border-color: var(--ui-border);
    border-left: 5px solid var(--ui-primary);
}
.step-chip {
    background: #F4F0EB;
    border-color: #DCD3CA;
    color: var(--ui-muted) !important;
}
.step-chip.done {
    background: #E3EFE8;
    border-color: #B9D2C3;
    color: var(--ui-success) !important;
}
.step-chip.active {
    background: var(--ui-primary);
    border-color: var(--ui-primary);
    color: #FFFFFF !important;
    box-shadow: 0 5px 14px rgba(198, 93, 50, 0.25);
}
.step-hero h1,
.block-container h1,
.block-container h2,
.block-container h3 {
    color: var(--ui-ink) !important;
}
.step-hero p,
.block-container [data-testid="stCaptionContainer"] p {
    color: var(--ui-muted) !important;
}
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stForm"] {
    background: rgba(255, 253, 249, 0.92);
    border-color: var(--ui-border) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #C6A98F !important;
    box-shadow: 0 11px 28px rgba(79, 62, 48, 0.10);
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div {
    background-color: var(--ui-surface) !important;
    border-color: #D4C7BA !important;
    color: var(--ui-ink) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--ui-primary) !important;
    box-shadow: 0 0 0 1px var(--ui-primary) !important;
}
.stSlider [role="slider"] {
    background-color: var(--ui-primary) !important;
    border-color: #FFFFFF !important;
}
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--ui-primary), #D4774E) !important;
    box-shadow: 0 6px 16px rgba(198, 93, 50, 0.22) !important;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--ui-primary-hover) !important;
}
div.stButton > button:disabled,
div[data-testid="stFormSubmitButton"] > button:disabled {
    background: #C9C4BE !important;
    color: #F7F5F2 !important;
    box-shadow: none !important;
}
[data-testid="stFileUploader"] button {
    background-color: var(--ui-secondary) !important;
}
div[data-testid="stAlertContainer"][data-baseweb="notification"] {
    border-radius: 11px !important;
}
div.stSuccess {
    background-color: #E8F2EC !important;
    border: 1px solid #B7D1C1 !important;
}
div.stSuccess * {
    color: #315F49 !important;
}
div.stInfo {
    background-color: #E8F0F2 !important;
    border-color: #B9CFD3 !important;
}
div.stInfo * {
    color: #365F67 !important;
}
div.stWarning {
    background-color: #FFF3E2 !important;
    border-color: #E9C796 !important;
}
div.stWarning * {
    color: #8A5A19 !important;
}
div.stError {
    background-color: #FBEAEA !important;
    border: 1px solid #E5B7B7 !important;
}
div.stError * {
    color: #8C3838 !important;
}
</style>
"""
)


def render_step_header(step: int, title: str, description: str) -> None:
    """Render one compact, consistent progress header."""
    labels = ("입력", "이미지 생성", "카피 합성", "완료")
    chips = []
    for index, label in enumerate(labels, start=1):
        state = "active" if index == step else "done" if index < step else ""
        chips.append(
            f'<span class="step-chip {state}">{index:02d} · {label}</span>'
        )
    st.html(
        '<section class="step-hero">'
        f'<div class="step-track">{"".join(chips)}</div>'
        f'<h1>{title}</h1>'
        f'<p>{description}</p>'
        '</section>'
    )

# 3. 입출력 경로 및 폴더 설정
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs/test_run")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4. 세션 상태 초기화 (현재 단계 기억)
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid4().hex

SESSION_INPUT_DIR = INPUT_DIR / st.session_state.session_id
SESSION_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# 5. 이미지 생성 전용 사이드바 설정 구역
saved_generation_settings = st.session_state.get("generation_settings", {})
gpt_model = saved_generation_settings.get("gpt_model", "gpt-5.4-nano")
layout_mode = saved_generation_settings.get("layout_mode", "layout")
product_focus = float(saved_generation_settings.get("product_focus", 1.0))
brand_focus = float(saved_generation_settings.get("brand_focus", 0.5))
seed = int(saved_generation_settings.get("seed", 42))

with st.sidebar:
    if st.session_state.step == 1:
        st.header("⚙️ 광고 생성 설정")
        if inference_queue.model_status == "loading":
            st.caption("🟡 GPU 모델을 백그라운드에서 준비하고 있습니다.")
        elif inference_queue.model_status == "ready":
            st.caption("🟢 GPU 모델 준비 완료")
        elif inference_queue.model_status == "failed":
            st.error(
                "GPU 모델 준비 실패: "
                f"{inference_queue.model_error}"
            )
        gpt_model = st.selectbox(
            "GPT 모델 선택", ["gpt-5.4-nano", "gpt-5.4-mini"]
        )
        layout_mode = st.radio(
            "레이아웃 모드", ["layout", "preserve"]
        )
        product_focus = st.slider(
            "상품 강조 강도 (Product Focus)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )
        brand_focus = st.slider(
            "배경 연출 강도 (Brand Focus)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help=(
                "0에 가까울수록 자연스러운 일상 배경, "
                "1에 가까울수록 고급 스튜디오 배경으로 연출합니다."
            ),
        )
        st.caption(
            "0 = 자연스러운 일상 배경 · 1 = 고급 스튜디오 배경"
        )
        seed = st.number_input("Seed 값", value=42, step=1)
    else:
        st.header("📍 진행 상태")
        st.caption(f"현재 STEP {st.session_state.step} / 4")
        st.caption("이미지 생성 설정이 확정되어 생성 옵션을 숨겼습니다.")

# --- STEP 1: 정보 및 파일 업로드 단계 ---
if st.session_state.step == 1:
    render_step_header(
        1,
        "상품 사진과 광고 정보를 입력해주세요",
        "입력 정보는 이미지 생성과 카피라이팅의 공통 기준으로 사용됩니다.",
    )

    col_image, col_product, col_store, col_ad = st.columns(
        [0.9, 1.0, 1.0, 1.0],
        gap="large",
    )

    with col_image, st.container(border=True):
        st.subheader("A. 상품 이미지")
        uploaded_image = st.file_uploader(
            "이미지 파일 (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_image:
            st.image(uploaded_image, caption="미리보기", width=280)

    with col_product, st.container(border=True):
        st.subheader("B. 상품 정보")
        product_name = st.text_input(
            "상품명", value="수제 베이커리 세트"
        )
        product_description = st.text_area(
            "상품 설명",
            value="다양한 빵과 쿠키, 아이스커피로 구성된 세트",
            height=96,
        )
        seller_description = st.text_area(
            "판매자 설명",
            value="매일 아침 직접 굽는 동네 베이커리",
            height=80,
        )

    with col_store, st.container(border=True):
        st.subheader("C. 매장·후기")
        store_info = st.text_input(
            "매장 정보", value="따뜻하고 편안한 분위기의 소규모 카페"
        )
        reviews_input = st.text_area(
            "리뷰 (엔터로 구분)",
            value="빵이 부드럽고 신선해요\n커피와 함께 먹기 좋아요",
            height=176,
        )

    with col_ad, st.container(border=True):
        st.subheader("D. 광고 방향")
        focus = st.selectbox(
            "광고 포커스",
            options=["product", "brand", "sales"],
            format_func=lambda value: {
                "product": "상품 중심",
                "brand": "브랜드 중심",
                "sales": "판매 전환 중심",
            }[value],
        )
        additional_request = st.text_area(
            "추가 요청사항",
            value="따뜻한 아침 햇살이 들어오는 카페 분위기",
            height=176,
        )

    st.divider()
    
    if st.button(
        "광고 제작 시작하기 ➔",
        use_container_width=True,
        type="primary",
    ):
        if uploaded_image:
            st.session_state.generation_settings = {
                "gpt_model": gpt_model,
                "layout_mode": layout_mode,
                "product_focus": float(product_focus),
                "brand_focus": float(brand_focus),
                "seed": int(seed),
            }
            # 1. 이미지 파일을 사용자 세션 폴더에 보존
            img_path = (
                SESSION_INPUT_DIR / Path(uploaded_image.name).name
            )
            
            with open(img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # 2. 입력받은 데이터들을 딕셔너리로 묶고 리스트 형태 정리
            reviews_list = [r.strip() for r in reviews_input.split('\n') if r.strip()]
                
            info_data = {
                "product_name": product_name,
                "product_description": product_description,
                "seller_description": seller_description,
                "store_info": store_info,
                "reviews": reviews_list,
                "focus": focus,
                "additional_request": additional_request
            }
                
            # 3. 딕셔너리를 JSON 파일로 자동 생성하여 저장
            json_path = SESSION_INPUT_DIR / "user_input_info.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info_data, f, ensure_ascii=False, indent=4)
                    
            # 다음 단계를 위해 세션 상태에 저장
            st.session_state.img_path = img_path
            st.session_state.json_path = json_path

            # 카피/레이아웃 합성을 제외한 이미지 작업만 FIFO 큐에 등록
            job_id = uuid4().hex
            job_output_dir = OUTPUT_DIR / job_id
            submission = inference_queue.submit(
                job_id=job_id,
                pipeline_kwargs={
                    "image_path": img_path,
                    "info_path": json_path,
                    "output_dir": job_output_dir,
                    "gpt_model": gpt_model,
                    "product_focus": float(product_focus),
                    "brand_focus": float(brand_focus),
                    "layout_mode": layout_mode,
                    "seed": int(seed),
                    "cpu_offload": False,
                },
            )
            st.session_state.job_id = job_id
            st.session_state.job_output_dir = job_output_dir
            st.session_state.pipeline_submission = submission
            st.session_state.step = 2
            st.rerun()
        else:
            st.warning("⚠️ 상품 이미지 파일을 업로드해야 실행할 수 있습니다.")

# --- STEP 2: 파이프라인 구동 단계 (VRAM 연산) ---
elif st.session_state.step == 2:
    render_step_header(
        2,
        "광고 이미지를 생성하고 있습니다",
        "장면 구성, 상품 복원, 경계 보정을 순서대로 처리합니다.",
    )
    
    with st.spinner("이미지 분석, 생성 및 상품 복원 중..."):
        try:
            submission = st.session_state.get(
                "pipeline_submission"
            )

            if submission is None:
                st.error("등록된 작업 정보를 찾을 수 없습니다.")
                if st.button("⬅ 처음으로 돌아가기"):
                    st.session_state.step = 1
                    st.rerun()
                st.stop()

            st.caption(f"작업 ID: `{submission.job_id}`")

            if not submission.future.done():
                if inference_queue.model_status == "loading":
                    st.info(
                        "GPU 모델을 준비하고 있습니다. "
                        "준비가 끝나면 작업을 자동으로 시작합니다."
                    )
                elif submission.future.running():
                    st.info("GPU Worker가 이미지를 생성하고 있습니다.")
                else:
                    st.info(
                        f"작업 대기 중입니다. 등록 시점에 "
                        f"앞에 {submission.queued_ahead}개의 작업이 "
                        "있었습니다."
                    )

                time.sleep(1)
                st.rerun()

            result = submission.future.result()
            st.session_state.pop("pipeline_submission", None)

            # 카피 합성 전, identity restoration 완료 이미지
            st.session_state.identity_image_path = Path(
                result.identity_restored_image
            )
            st.session_state.prompt_json_path = Path(
                result.prompt_json
            )
            
            st.session_state.step = 3
            st.rerun()
            
        except Exception as e:
            st.session_state.pop("pipeline_submission", None)
            st.error(f"파이프라인 실행 중 오류가 발생했습니다: {e}")
            if st.button("⬅ 처음으로 돌아가기"):
                st.session_state.step = 1
                st.rerun()

# --- STEP 3: 카피라이팅 입력 및 합성 단계 ---
elif st.session_state.step == 3:
    render_step_header(
        3,
        "광고 문구와 레이아웃을 완성해주세요",
        "문구 톤과 길이를 선택하면 완성된 이미지 위에 카피를 합성합니다.",
    )

    identity_img = st.session_state.get("identity_image_path")
    if identity_img is not None:
        identity_img = Path(identity_img)

    if identity_img is None or not identity_img.exists():
        result_dir = Path(
            st.session_state.get("job_output_dir", OUTPUT_DIR)
        ) / "05_final"
        output_files = (
            list(result_dir.rglob("*.png"))
            + list(result_dir.rglob("*.jpg"))
            + list(result_dir.rglob("*.jpeg"))
        )
        identity_img = (
            max(output_files, key=os.getmtime)
            if output_files
            else None
        )

    identity_ready = identity_img is not None and identity_img.is_file()
    if identity_ready:
        st.success("✅ 이미지 생성이 완료됐습니다. 카피 설정 후 합성을 시작하세요.")
    else:
        st.error(
            "카피를 합성할 이미지를 찾지 못했습니다. "
            "진행 버튼을 사용할 수 없으므로 새로운 광고를 생성해주세요."
        )

    with st.form("copywriting_form"):
        tone_col, length_col, action_col = st.columns(
            [1.4, 0.8, 1.3],
            gap="large",
            vertical_alignment="bottom",
        )
        with tone_col:
            copy_tone = st.text_input(
                "문구 톤",
                value="따뜻하고 친근한",
                help="예: 전문적이고 신뢰감 있는, 밝고 재치 있는",
            )
        with length_col:
            copy_length = st.selectbox(
                "글 길이",
                options=["short", "medium", "long"],
                format_func=lambda value: {
                    "short": "짧게",
                    "medium": "보통",
                    "long": "길게",
                }[value],
            )
        with action_col:
            create_copy = st.form_submit_button(
                "STEP 4: 카피 생성 및 합성 ➔",
                use_container_width=True,
                type="primary",
                disabled=not identity_ready,
            )

    if identity_ready:
        with st.expander("합성 전 이미지 확인", expanded=False):
            st.image(
                str(identity_img),
                width=320,
                caption="Copy layout 전 이미지",
            )

    if create_copy and identity_ready:
        copy_tone = copy_tone.strip()
        if not copy_tone:
            st.error("문구 톤을 입력해주세요.")
        else:
            try:
                from adcg.pipelines.copy_layout import (
                    run_copy_layout_pipeline,
                )

                prompt_json = Path(
                    st.session_state.get(
                        "prompt_json_path",
                        Path(st.session_state.job_output_dir)
                        / "02_prompt"
                        / "ad_prompt.json",
                    )
                )

                with st.spinner("광고 문구 생성과 레이아웃 합성 중..."):
                    copy_result = run_copy_layout_pipeline(
                        identity_image=identity_img,
                        info_path=st.session_state.json_path,
                        prompt_json=prompt_json,
                        output_dir=st.session_state.job_output_dir,
                        gpt_model=gpt_model,
                        copy_count=1,
                        copy_tone=copy_tone,
                        copy_length=copy_length,
                    )

                st.session_state.copy_json_path = Path(
                    copy_result.copy_json
                )
                st.session_state.final_image_path = Path(
                    copy_result.final_image
                )
                st.session_state.step = 4
                st.rerun()
            except Exception as e:
                traceback.print_exc()
                copy_json_candidate = (
                    Path(st.session_state.job_output_dir)
                    / "02_prompt"
                    / "ad_copy.json"
                )
                if copy_json_candidate.is_file():
                    st.warning(
                        "광고 문구 생성은 완료됐지만 레이아웃 합성에서 "
                        "중단됐습니다. VM 터미널의 상세 로그를 확인해주세요."
                    )
                else:
                    st.warning(
                        "광고 문구 생성 단계에서 중단됐습니다. "
                        "OpenAI 설정과 VM 터미널 로그를 확인해주세요."
                    )
                st.exception(e)

    st.divider()
    if st.button("🔄 새로운 광고 만들기", use_container_width=True):
        st.session_state.step = 1
        for key in (
            "identity_image_path",
            "prompt_json_path",
            "copy_json_path",
            "final_image_path",
            "job_output_dir",
            "job_id",
            "pipeline_submission",
            "generation_settings",
        ):
            st.session_state.pop(key, None)
        st.rerun()

# --- STEP 4: 카피 합성 결과 표출 단계 ---
elif st.session_state.step == 4:
    render_step_header(
        4,
        "완성된 광고 시안",
        "결과를 확인하고 다운로드하거나 카피를 다시 생성할 수 있습니다.",
    )

    final_img = st.session_state.get("final_image_path")
    if final_img is not None:
        final_img = Path(final_img)

    copy_data = {}
    copy_json_path = st.session_state.get("copy_json_path")
    if copy_json_path is not None and Path(copy_json_path).exists():
        copy_document = json.loads(
            Path(copy_json_path).read_text(encoding="utf-8")
        )
        copies = copy_document.get("copies", [])
        if copies:
            copy_data = copies[0]

    if final_img is not None and final_img.exists():
        col_image, col_info = st.columns([6, 4])

        with col_image, st.container(border=True):
            st.image(
                str(final_img),
                use_column_width=True,
                caption="Copy layout 완료 이미지",
            )

        with col_info, st.container(border=True):
            st.success("✅ 카피와 레이아웃 합성이 완료됐습니다.")
            if copy_data:
                st.subheader("생성된 광고 문구")
                st.markdown(f"### {copy_data.get('title', '')}")
                if copy_data.get("subtitle"):
                    st.write(copy_data["subtitle"])
                if copy_data.get("price"):
                    st.write(f"가격: {copy_data['price']}")
                if copy_data.get("cta"):
                    st.write(copy_data["cta"])

            st.info(f"📁 **로컬 저장소 경로:**\n{final_img.absolute()}")

            with open(final_img, "rb") as file:
                st.download_button(
                    label="광고 이미지 다운로드 📥",
                    data=file,
                    file_name=final_img.name,
                    mime="image/png",
                    use_container_width=True,
                )
    else:
        st.error("최종 카피 합성 이미지를 찾지 못했습니다.")

    st.divider()
    col_retry, col_restart = st.columns(2)
    with col_retry:
        if st.button("✏️ 카피 다시 만들기", use_container_width=True):
            st.session_state.step = 3
            st.session_state.pop("copy_json_path", None)
            st.session_state.pop("final_image_path", None)
            st.rerun()
    with col_restart:
        if st.button("🔄 새로운 광고 만들기", use_container_width=True):
            st.session_state.step = 1
            for key in (
                "identity_image_path",
                "prompt_json_path",
                "copy_json_path",
                "final_image_path",
                "job_output_dir",
                "job_id",
                "pipeline_submission",
                "generation_settings",
            ):
                st.session_state.pop(key, None)
            st.rerun()
