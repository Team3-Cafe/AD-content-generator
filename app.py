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


# ---------------------------------------------------------
# 1. 페이지 기본 설정 및 통합 커스텀 CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 로 만드는 우리 가게 광고",
    layout="wide",
)

st.html("""
<style>
    /* 1. 사이드바 영역 및 접기/펼치기 화살표 버튼 완전 제거 */
    [data-testid="stSidebar"], 
    [data-testid="collapsedControl"],
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* 2. 전체 앱 및 배경 (웜 화이트 & 소프트 베이지) */
    .stApp {
        background: radial-gradient(circle at 92% 2%, rgba(211, 84, 0, 0.10), transparent 28rem), 
                    linear-gradient(145deg, #FDFBF7 0%, #F8F2EA 100%);
    }

    /* 3. 메인 컨테이너 여백 최적화 */
    .block-container {
        max-width: 1600px;
        padding-top: 1.25rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
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

    .step-hero {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(196, 169, 143, 0.50);
        border-radius: 18px;
        box-shadow: 0 14px 40px rgba(89, 63, 42, 0.08);
        padding: 1rem 1.25rem 1.05rem;
        margin: 0 0 0.7rem;
        backdrop-filter: blur(10px);
    }
</style>
""")

# ---------------------------------------------------------
# 2. 리소스 / 파이프라인 / 큐 초기화
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_inference_queue():
    """Keep dedicated Dual and Canny pipelines on one FIFO GPU worker."""

    def load_background_pipe():
        return load_generation_pipeline(
            base_model=GENERATION_DEFAULTS["base_model"],
            controlnet_model=GENERATION_DEFAULTS["controlnet_model"],
            depth_controlnet_model=GENERATION_DEFAULTS[
                "depth_controlnet_model"
            ],
            cpu_offload=False,
        )

    def load_identity_pipe():
        return load_generation_pipeline(
            base_model=GENERATION_DEFAULTS["base_model"],
            controlnet_model=GENERATION_DEFAULTS["controlnet_model"],
            depth_controlnet_model=None,
            cpu_offload=False,
        )

    return InferenceQueue(
        background_pipe_factory=load_background_pipe,
        identity_pipe_factory=load_identity_pipe,
    )


# Streamlit rerun 중에도 같은 작업 큐와 두 파이프라인을 재사용한다.
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
    background: #2A3A30;
    border-color: #2A3A30;
    color: #FFFFFF !important;
    box-shadow: 0 5px 14px rgba(42, 58, 48, 0.24);
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
h1, h2, h3, h4, h5, h6, p, label {
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
    border-left: 1px solid var(--ui-border);
    box-sizing: border-box;
    overflow: hidden;
    position: relative;
}
.step-hero::before {
    background: var(--ui-primary);
    content: "";
    inset: 0 auto 0 0;
    position: absolute;
    width: 5px;
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
    border: 1px solid #D4C7BA !important;
    color: var(--ui-ink) !important;
}
[data-testid="stTextArea"] textarea {
    line-height: 1.45 !important;
    resize: none !important;
}
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: var(--ui-surface) !important;
    border: 1px solid #D4C7BA !important;
    border-radius: 9px !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: var(--ui-ink) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border: 0 !important;
    box-shadow: none !important;
    outline: 0 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--ui-primary) !important;
    box-shadow: 0 0 0 1px var(--ui-primary) !important;
}
[data-testid="stWidgetLabel"] {
    margin-bottom: 0.24rem !important;
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

/* Sidebar: keep the light visual system across BaseWeb controls. */
section[data-testid="stSidebar"] {
    min-width: 340px !important;
    width: 340px !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: 340px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: #FFFDF9 !important;
    border-color: #C8D4D5 !important;
    color: var(--ui-ink) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input {
    color: var(--ui-ink) !important;
    -webkit-text-fill-color: var(--ui-ink) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
[data-testid="stSidebar"] [data-baseweb="base-input"]:focus-within {
    border-color: var(--ui-primary) !important;
}
.st-key-sidebar_model_group,
.st-key-sidebar_layout_group,
.st-key-sidebar_focus_group {
    background: rgba(255, 253, 249, 0.72);
    border: 1px solid #CAD7D8;
    border-radius: 14px;
    box-sizing: border-box;
    margin-bottom: 0.55rem;
    padding: 0.8rem 0.85rem 0.9rem;
}
.sidebar-section-title {
    color: var(--ui-secondary) !important;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    margin: 0 0 0.45rem;
}
.model-status {
    align-items: center;
    background: rgba(255,255,255,0.62);
    border: 1px solid #C9D7D9;
    border-radius: 999px;
    color: #4F6167 !important;
    display: flex;
    font-size: 0.77rem;
    gap: 0.45rem;
    line-height: 1.3;
    margin: 0.2rem 0 0.8rem;
    padding: 0.48rem 0.68rem;
}
.model-status-dot {
    animation: processingPulse 1.6s ease-in-out infinite;
    background: #D5A545;
    border-radius: 999px;
    flex: 0 0 auto;
    height: 9px;
    width: 9px;
}
.model-status.ready .model-status-dot {
    animation: none;
    background: #55A77A;
}
.focus-scale {
    color: #64757A !important;
    display: flex;
    font-size: 0.68rem;
    justify-content: space-between;
    margin: -0.15rem 0 0.25rem;
}
.focus-description {
    background: #EEF4F3;
    border-radius: 9px;
    color: #53676C !important;
    font-size: 0.7rem;
    line-height: 1.45;
    margin: 0.2rem 0 0;
    padding: 0.52rem 0.6rem;
}
.st-key-layout_mode_control [role="radiogroup"] {
    display: grid !important;
    gap: 0.4rem !important;
    grid-template-columns: 1fr 1fr;
}
.st-key-layout_mode_control [role="radiogroup"] label {
    background: #FFFDF9;
    border: 1px solid #C8D4D5;
    border-radius: 9px;
    box-sizing: border-box;
    justify-content: center;
    margin: 0 !important;
    min-height: 38px;
    padding: 0.42rem 0.5rem;
}
.st-key-layout_mode_control [role="radiogroup"] label > div:first-child {
    display: none;
}
.st-key-layout_mode_control [role="radiogroup"] label:has(input:checked) {
    background: var(--ui-secondary);
    border-color: var(--ui-secondary);
}
.st-key-layout_mode_control [role="radiogroup"] label:has(input:checked) p {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255, 253, 249, 0.58);
    border: 1px solid #CAD7D8;
    border-radius: 12px;
    overflow: hidden;
}

/* STEP 3: preview and copy controls use purpose-built cards. */
.st-key-copy_preview_panel,
.st-key-copy_settings_panel,
.st-key-final_preview_panel,
.st-key-final_details_panel {
    background: rgba(255, 253, 249, 0.96);
    border: 1px solid #D8C9BA;
    border-radius: 18px;
    box-sizing: border-box;
    box-shadow: 0 12px 30px rgba(57, 45, 36, 0.09);
    overflow: hidden;
    padding: 1.15rem 1.2rem 1.25rem;
}
.st-key-copy_preview_panel {
    background:
        linear-gradient(145deg, rgba(238, 244, 243, 0.96), rgba(255, 253, 249, 0.98));
    border-color: #C8D8D7;
}
.st-key-copy_preview_panel [data-testid="stImage"] {
    display: flex;
    justify-content: flex-start;
}
.st-key-copy_preview_panel [data-testid="stImage"] img {
    border: 1px solid #CBD6D5;
    border-radius: 14px;
    box-shadow: 0 10px 26px rgba(36, 50, 59, 0.14);
}
.copy-panel-kicker {
    color: var(--ui-primary) !important;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    margin: 0 0 0.18rem;
}
.copy-panel-title {
    color: var(--ui-ink) !important;
    font-size: 1.18rem;
    font-weight: 800;
    line-height: 1.35;
    margin: 0 0 0.35rem;
}
.copy-panel-help {
    color: var(--ui-muted) !important;
    font-size: 0.84rem;
    line-height: 1.55;
    margin: 0 0 0.8rem;
}
.copy-ready-note {
    align-items: center;
    background: #E9F2ED;
    border: 1px solid #BED5C7;
    border-radius: 12px;
    color: #315F49 !important;
    display: flex;
    font-size: 0.84rem;
    font-weight: 700;
    gap: 0.45rem;
    margin: 0 0 0.85rem;
    padding: 0.68rem 0.8rem;
}
.copy-process {
    background: #F6F0E9;
    border: 1px solid #E2D3C4;
    border-radius: 12px;
    color: #65584D !important;
    font-size: 0.79rem;
    line-height: 1.5;
    margin: 0.75rem 0 0.85rem;
    padding: 0.7rem 0.8rem;
}
.copy-process strong {
    color: var(--ui-primary) !important;
}
.st-key-copy_settings_panel [data-testid="stForm"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.processing-panel {
    background: linear-gradient(120deg, #263B43, #3F6B73);
    border: 1px solid #527B82;
    border-radius: 16px;
    box-shadow: 0 12px 28px rgba(36, 50, 59, 0.20);
    box-sizing: border-box;
    color: #FFFFFF !important;
    margin: 0 0 0.9rem;
    overflow: hidden;
    padding: 1rem 1.1rem 0.9rem;
    position: relative;
}
.processing-panel::after {
    animation: processingSweep 2.2s ease-in-out infinite;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
    content: "";
    height: 100%;
    left: -45%;
    position: absolute;
    top: 0;
    width: 38%;
}
.processing-heading {
    align-items: center;
    display: flex;
    gap: 0.65rem;
    position: relative;
    z-index: 1;
}
.processing-orb {
    animation: processingPulse 1.15s ease-in-out infinite;
    background: #F3B56F;
    border: 3px solid rgba(255,255,255,0.28);
    border-radius: 999px;
    box-shadow: 0 0 0 0 rgba(243,181,111,0.5);
    flex: 0 0 auto;
    height: 13px;
    width: 13px;
}
.processing-title,
.processing-description,
.processing-steps,
.processing-steps span {
    color: #FFFFFF !important;
    position: relative;
    z-index: 1;
}
.processing-title {
    font-size: 1rem;
    font-weight: 800;
}
.processing-description {
    font-size: 0.81rem;
    margin: 0.35rem 0 0.7rem 1.8rem;
    opacity: 0.82;
}
.processing-steps {
    display: flex;
    flex-wrap: wrap;
    gap: 0.38rem;
    margin-left: 1.8rem;
}
.processing-steps span {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 999px;
    font-size: 0.7rem;
    padding: 0.24rem 0.52rem;
}
.st-key-generation_progress_region,
.st-key-copy_progress_detail {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.st-key-final_preview_panel [data-testid="stImage"] {
    display: flex;
    justify-content: center;
}
.st-key-final_preview_panel [data-testid="stImage"] img {
    border: 1px solid #CBD6D5;
    border-radius: 12px;
    box-shadow: 0 10px 24px rgba(36, 50, 59, 0.12);
}
[data-testid="stFileUploaderDropzone"],
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    box-sizing: border-box;
}
[data-testid="stFileUploaderDropzone"] {
    overflow: hidden;
}
@keyframes processingPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(243,181,111,0.48); opacity: 0.72; }
    50% { box-shadow: 0 0 0 9px rgba(243,181,111,0); opacity: 1; }
}
@keyframes processingSweep {
    0% { left: -45%; }
    60%, 100% { left: 115%; }
}
@media (max-width: 900px) {
    .st-key-copy_preview_panel,
    .st-key-copy_settings_panel,
    .st-key-final_preview_panel,
    .st-key-final_details_panel {
        padding: 0.9rem;
    }
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


def render_processing_panel(
    title: str,
    description: str,
    steps: tuple[str, ...],
) -> None:
    """Render the same animated processing state for generation and copy."""
    step_items = "".join(
        f"<span>{index:02d} · {label}</span>"
        for index, label in enumerate(steps, start=1)
    )
    st.html(
        '<section class="processing-panel">'
        '<div class="processing-heading">'
        '<span class="processing-orb"></span>'
        f'<strong class="processing-title">{title}</strong>'
        '</div>'
        f'<p class="processing-description">{description}</p>'
        f'<div class="processing-steps">{step_items}</div>'
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


# 화면 비율별 프리셋 설정 (미로 보드 데이터 기준)
ASPECT_RATIO_PRESETS = {
    "1:1": {
        "label": "1:1 (정사각형 피드 - 1080×1080)",
        "width": 1080,
        "height": 1080,
    },
    "3:4": {
        "label": "3:4 (세로형 카드뉴스 - 768×1024)",
        "width": 768,
        "height": 1024,
    },
    "9:16": {
        "label": "9:16 (세로형 풀스크린 / 숏폼 - 720×1280)",
        "width": 720,
        "height": 1280,
    },
    "16:9": {
        "label": "16:9 (가로형 와이드 / PPT - 1280×720)",
        "width": 1280,
        "height": 720,
    },
    "4:3": {
        "label": "4:3 (가로형 표준 / 태블릿 - 1024×768)",
        "width": 1024,
        "height": 768,
    },
}

# =========================================================
# 👇 [여기에 3단계 페이지 전환 로직을 삽입하세요!] 👇
# =========================================================
# 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 1

# 상단 진행 상태 헤더 (Step Indicator)
st.html(f"""
<div style="display: flex; gap: 10px; margin-bottom: 20px;">
    <span style="padding: 6px 16px; border-radius: 20px; font-weight: bold; background: {'#2A3A30' if st.session_state.step == 1 else '#E0E0E0'}; color: {'white' if st.session_state.step == 1 else '#666'};">01. 기본 정보 입력</span>
    <span style="padding: 6px 16px; border-radius: 20px; font-weight: bold; background: {'#2A3A30' if st.session_state.step == 2 else '#E0E0E0'}; color: {'white' if st.session_state.step == 2 else '#666'};">02. 레이아웃 & 포커스 설정</span>
    <span style="padding: 6px 16px; border-radius: 20px; font-weight: bold; background: {'#2A3A30' if st.session_state.step == 3 else '#E0E0E0'}; color: {'white' if st.session_state.step == 3 else '#666'};">03. 광고 생성 & 결과</span>
</div>
""")

# -------------------------------------------------------------------
# [STEP 01] PPT 슬라이드 31: 상품 사진 및 매장/기본 정보 입력
# -------------------------------------------------------------------
if st.session_state.step == 1:
    # [1] 더 강력해진 UI 스타일 커스텀 CSS 주입
    st.html("""
    <style>
    /* 라벨(입력칸 위 제목) 크기 확대 */
    .stTextInput label p, .stTextArea label p, .stRadio label p, .stFileUploader label p {
    font-size: 17px !important;
    }

    /* 1. 기본 상태 및 작성 전: 흰색 배경 + 어두운 회색 테두리 */
    .stTextInput input, .stTextArea textarea {
        background-color: white !important;
        color: black !important;
        border: 2px solid #555555 !important;  /* 어두운 회색 테두리 */
        border-radius: 8px !important;
        font-size: 17px !important;
        padding: 10px !important;
    }

    /* 2. 클릭(포커스)하거나 사용자가 입력 중일 때: 어두운 회색으로 변경 */
    .stTextInput input:focus, .stTextArea textarea:focus {
        background-color: white !important;
        color: black !important;
        border: 2px solid #555555 !important;  /* 어두운 회색 테두리 */
        border-radius: 10px !important;
        box-shadow: 0 0 0 1px #E26848 !important;
    }

    /* 3. 값이 채워진 후(입력 완료 상태): 부드러운 배경 및 테두리 처리 */
    .stTextInput input:not(:placeholder-shown), .stTextArea textarea:not(:placeholder-shown) {
        border: 2px solid #555555 !important;  /* 평소에는 어두운 회색 라인 유지 */
    }

    /* Placeholder(예시 글자) 색상 */
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #888888 !important;
    }
    
    /* 4. 파일 업로드 중앙 아이콘 🔄로 변경 (중복 생성 방지) */
    [data-testid="stFileUploadDropzone"] div svg {
        display: none !important; 
    }
    [data-testid="stFileUploadDropzone"] div:first-child::before {
        content: "🔄";
        font-size: 28px;
        display: block;
        text-align: center;
        margin-bottom: 5px;
    }
    
    /* 5. 업로드 완료 후 나타나는 우측 삭제(X/+) 버튼을 🔄 로 교체 */
    [data-testid="stFileUploaderDeleteBtn"] svg {
        display: none !important;
    }
    [data-testid="stFileUploaderDeleteBtn"]::before {
        content: "🔄";
        font-size: 16px;
        display: block;
        color: #555555;
    }
    /* 👇 사이드바 토글 버튼(>>) 숨기기 */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
    """)

if st.session_state.step == 1:
    # [2] 제목과 소제목을 직접 HTML로 작성하여 색상/크기 절대 보장
    st.html('<h1 style="color: #E26848; font-size: 3.2rem; font-weight: bold; margin-bottom: 0;">STEP 01. 사장님의 상품과 매장을 알려주세요</h1>')
    st.html('<p style="font-size: 1.1rem; color: #888888; margin-bottom: 2rem;">사진 한 장과 매장 정보만 있으면 AI가 나머지를 그립니다.</p>')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.html('<h3 style="color: #E26848; font-size: 1.8rem; margin-top: 1rem; margin-bottom: 1rem;">A. 상품·서비스 사진 업로드</h3>')
        uploaded_file = st.file_uploader("이미지 선택 (JPG, PNG)", type=["jpg", "png", "jpeg"])
        
        # 👇 업로드된 파일을 세션 상태에 저장하여 다음 단계에서도 유지되도록 합니다.
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
        
        # 👇 안내 문구
        st.caption("📁 200MB 이하의 jpg, png 파일로 업로드해주세요.")
        st.caption("🔄 새로운 사진을 올리면 기존 사진이 자동으로 변경됩니다.")
        
        # 👇 기존 우측에 있던 B 섹션을 좌측 A 아래로 이동
        st.html('<h3 style="color: #E26848; font-size: 1.8rem; margin-top: 1rem; margin-bottom: 1rem;">B. 매장 정보 & 타겟 설정</h3>')
        biz_type = st.radio("사업 유형", ["D2C (매장 방문형)", "B2C (출장 서비스형)"], horizontal=True)

        # 선택된 사업 유형에 따라 라벨과 예시 문구를 다르게 설정합니다.
        if biz_type == "B2C (출장 서비스형)":
            name_label = "사업자명"
            name_placeholder = "예: 꼼꼼 홈클리닝"
            loc_label = "출장 반경"
            loc_placeholder = "예: 강원도 원주시 전 지역"
        else:
            name_label = "매장명"
            name_placeholder = "예: 천사로 베이커리 카페"
            loc_label = "매장 위치"
            loc_placeholder = "예: 강원도 원주시 천사로 1004"

        ##예시 자동 입력 버튼 ----------------------------------------------
        
        # 위에서 설정한 변수를 text_input에 적용합니다.
        store_name = st.text_input(name_label, placeholder=name_placeholder)
        store_location = st.text_input(loc_label, placeholder=loc_placeholder)

    with col2:
        # C 섹션
        st.html('<h3 style="color: #E26848; font-size: 1.8rem; margin-top: 1rem; margin-bottom: 1rem;">C. 홍보하기</h3>')
        
        product_name = st.text_input("홍보 대상", placeholder="예: 우리 동네 모닝 베이커리 세트")
        product_desc = st.text_area("업로드한 이미지를 단어로 친절하게 묘사해주세요.", placeholder="예: 오늘 아침 갓 구운 신선한 소금빵과 시원한 아이스 아메리카노")

        # 필수 항목 검사 및 다음 단계 버튼
        if st.button("다음 단계로 (레이아웃 & 포커스 설정) ➔", use_container_width=True):
            if not uploaded_file:
                st.warning("⚠️ 상품·서비스 사진을 업로드해주세요.")
            elif not product_name.strip():
                st.warning("⚠️ 홍보 대상을 입력해주세요.")
            elif not store_name.strip():
                st.warning(f"⚠️ {name_label}을(를) 입력해주세요.")
            elif not store_location.strip():
                st.warning(f"⚠️ {loc_label}을(를) 입력해주세요.")
            else:
                # 다음 단계와 파이프라인에서 사용할 사용자 입력을 보존한다.
                st.session_state.product_name = product_name.strip()
                st.session_state.product_description = product_desc.strip()
                st.session_state.store_name = store_name.strip()
                st.session_state.store_location = store_location.strip()
                st.session_state.store_type = biz_type
                st.session_state.step = 2
                st.rerun()

# -------------------------------------------------------------------
# [STEP 02] PPT 슬라이드 32: 레이아웃 구성, 화면 비율 & Focus 조율
# -------------------------------------------------------------------
# --- STEP 2: 레이아웃 및 포커스 설정 단계 ---
elif st.session_state.step == 2:
    render_step_header(2, "레이아웃과 포커스 설정", "구도와 강조점을 설정하여 원하는 스타일의 광고를 디자인합니다.")
    
    # 화면을 좌우 1:1 비율로 나눕니다. (왼쪽: 이미지 / 오른쪽: 설정)
    col1, col2 = st.columns([1, 1])
    
    # ---------------------------------------------------------
    # [좌측] 이미지 미리보기 영역
    # ---------------------------------------------------------
    with col1:
        st.html('<p style="color: #E26848; font-weight: bold; font-size: 14px; margin-bottom: 5px;">BASE IMAGE</p>')
        st.html('<h3 style="margin-top: 0;">작업 이미지 미리보기</h3>')
        st.caption("레이아웃을 적용할 원본 또는 누끼(배경 제거) 이미지입니다.")
        
        # STEP 1에서 넘어온 이미지를 띄웁니다.
        if "uploaded_file" in st.session_state and st.session_state.uploaded_file:
            st.image(st.session_state.uploaded_file, use_container_width=True)
        else:
            st.warning("미리보기를 표시할 이미지가 없습니다.")
            
    # ---------------------------------------------------------
    # [우측] 레이아웃 & 포커스 설정 영역 (사장님 코드 통합)
    # ---------------------------------------------------------
    with col2:
        st.html('<p style="color: #E26848; font-weight: bold; font-size: 14px; margin-bottom: 5px;">LAYOUT & FOCUS</p>')
        st.html('<h3 style="margin-top: 0;">레이아웃과 포커스 조율</h3>')
        st.caption("원하는 광고 스타일에 맞춰 구도와 강조점을 설정해 주세요.")
        
        # 1. LAYOUT & RATIO 설정
        st.html('<p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #31333F; margin-top: 15px;">레이아웃 구성</p>')
        layout_mode = st.radio(
            "레이아웃 구성",
            ["layout", "preserve"],
            format_func=lambda x: "새롭게 구성" if x == "layout" else "원본 유지",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.html('<p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #31333F; margin-top: 15px;">화면 비율</p>')
        aspect_ratio = st.selectbox(
            "화면 비율",
            ["1:1 (정사각형 피드)", "3:4 (카드뉴스)", "9:16 (숏폼/릴스)", "16:9 (가로 와이드)", "4:3 (태블릿)"],
            label_visibility="collapsed"
        )
        
        # 2. FOCUS & SCALE SLIDERS 설정 (기존 focus 유지 + scale 추가)
        st.html('<p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #31333F; margin-top: 25px;">상품 강조 정도</p>')
        # 1. 상품 강조 강도
        product_focus = st.slider(
            "상품 강조 강도", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help=(
                "0에 가까울수록 주변 배경과 부드럽게 어우러지며, "
                "1에 가까울수록 상품의 원래 형태와 디테일을 뚜렷하게 유지합니다."
            )
        )
        # 🚨 [새로 추가된 항목] 상품 크기 비율 (Product Scale) - 기본값 1.0 기준
        st.html('<p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #31333F; margin-top: 15px;">상품 크기 비율 (Scale)</p>')
        # 2. 상품 크기 (Scale)
        product_scale = st.slider(
            "상품 크기 (Scale)", 
            min_value=0.5, 
            max_value=1.5, 
            value=1.0, 
            step=0.1,
            help=(
                "0.5에 가까울수록 상품이 작아지며 주변 여백이 넓어지고, "
                "1.5에 가까울수록 상품이 화면을 꽉 채우도록 커집니다. (1.0 = 원본)"
            )
        )
        st.html('<p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #31333F; margin-top: 15px;">브랜드/배경 강조 정도</p>')
        # 3. 브랜드/배경 강조 강도
        brand_focus = st.slider(
            "배경 연출 강도", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help=(
                "0에 가까울수록 자연스러운 일상 배경, "
                "1에 가까울수록 고급 스튜디오 배경으로 연출합니다."
            )
        )
        # STEP 3와 통일된 시각적 디테일 (안내 박스)
        st.html("""
        <div style="background-color: #F8F9FA; padding: 15px; border-radius: 8px; margin-top: 25px; border: 1px solid #E9ECEF;">
            <p style="font-size: 13px; font-weight: bold; color: #E26848; margin-bottom: 5px;">다음 단계 안내</p>
            <p style="font-size: 13px; color: #555; margin: 0;">설정한 레이아웃과 포커스를 바탕으로 다음 단계에서 카피라이팅과 최종 합성을 진행합니다.</p>
        </div>
        """)
        
    st.html("<br><br>")  # 여백 추가
    
    # ---------------------------------------------------------
    # [하단] 액션 버튼 영역 (버튼 두 개 나란히 배치)
    # ---------------------------------------------------------
    btn_col1, btn_col2 = st.columns([1, 1])
    
    with btn_col1:
        if st.button("새로운 광고 만들기 🔄", key="step2_reset", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
            
    with btn_col2:
        if st.button("STEP 3: 카피 작성 단계로 이동 ➔", type="primary", use_container_width=True):
            # 1. 설정한 값들을 세션에 저장
            st.session_state.layout_mode = layout_mode
            st.session_state.aspect_ratio = aspect_ratio
            st.session_state.product_focus = product_focus
            st.session_state.product_scale = product_scale
            st.session_state.brand_focus = brand_focus

            # 2. 작업 ID별 입력/출력 경로를 만든다.
            job_id = uuid4().hex
            job_input_dir = SESSION_INPUT_DIR / job_id
            job_output_dir = OUTPUT_DIR / job_id
            job_input_dir.mkdir(parents=True, exist_ok=True)
            job_output_dir.mkdir(parents=True, exist_ok=True)

            uploaded = st.session_state.get("uploaded_file")
            if uploaded is None:
                st.error("업로드한 이미지가 없습니다. STEP 1부터 다시 진행해주세요.")
                st.stop()

            image_name = Path(
                getattr(uploaded, "name", "uploaded_image.png")
            ).name
            image_path = job_input_dir / image_name
            if isinstance(uploaded, (str, os.PathLike)):
                image_path = Path(uploaded)
            else:
                with open(image_path, "wb") as file:
                    file.write(uploaded.getbuffer())

            # 3. 장면 프롬프트와 카피에 공통으로 사용할 상품 정보 JSON.
            store_name = st.session_state.get("store_name", "")
            store_location = st.session_state.get("store_location", "")
            info_data = {
                "product_name": st.session_state.get("product_name", ""),
                "product_description": st.session_state.get(
                    "product_description",
                    "",
                ),
                "seller_description": store_name,
                "store_info": " / ".join(
                    value
                    for value in (store_name, store_location)
                    if value
                ),
                "reviews": [],
                "focus": "product",
                "additional_request": "",
                "store_name": store_name,
                "store_type": st.session_state.get("store_type", ""),
            }
            info_json_path = job_input_dir / "user_input_info.json"
            with open(info_json_path, "w", encoding="utf-8") as file:
                json.dump(info_data, file, ensure_ascii=False, indent=4)

            # 선택한 화면 비율에 맞춰 기존 생성 해상도를 사용한다.
            target_width, target_height = 1024, 1024
            if "3:4" in aspect_ratio:
                target_width, target_height = 768, 1024
            elif "9:16" in aspect_ratio:
                target_width, target_height = 576, 1024
            elif "16:9" in aspect_ratio:
                target_width, target_height = 1024, 576
            elif "4:3" in aspect_ratio:
                target_width, target_height = 1024, 768

            # 4. 준비 단계는 병렬로, Diffusers 추론은 공유 GPU 큐로 실행한다.
            submission = inference_queue.submit(
                job_id=job_id,
                pipeline_kwargs={
                    "image_path": image_path,
                    "info_path": info_json_path,
                    "output_dir": job_output_dir,
                    "layout_mode": layout_mode,
                    "width": target_width,
                    "height": target_height,
                    "product_focus": float(product_focus),
                    "product_scale": float(product_scale),
                    "brand_focus": float(brand_focus),
                    "cpu_offload": False,
                },
            )

            st.session_state.job_id = job_id
            st.session_state.job_output_dir = job_output_dir
            st.session_state.image_path = image_path
            st.session_state.prompt_json_path = info_json_path
            st.session_state.pipeline_submission = submission
            st.session_state.pop("background_image_path", None)
            st.session_state.pop("generated_prompt_json", None)
            st.session_state.step = 3
            st.rerun()



# --- STEP 3: 카피라이팅 입력 및 합성 단계 ---
if st.session_state.step == 3:
    submission = st.session_state.get("pipeline_submission")
    if submission is not None:
        render_step_header(
            3,
            "광고 이미지를 생성하고 있습니다",
            "상품 분석과 장면 구성이 끝나면 카피 작성 화면으로 이동합니다.",
        )
        render_processing_panel(
            "광고 이미지를 제작하고 있습니다.",
            "브라우저를 닫아도 작업은 GPU 대기열에서 계속 처리됩니다.",
            ("상품 분석", "장면 구성", "GPU 생성", "상품 복원"),
        )
        st.caption(f"작업 ID: `{submission.job_id}`")

        try:
            if not submission.future.done():
                if submission.stage == "preparing":
                    st.info(
                        "누끼 이미지와 장면 프롬프트를 만든 뒤 "
                        "Canny·Depth 조건을 준비하고 있습니다."
                    )
                elif inference_queue.model_status == "loading":
                    st.info(
                        "배경 생성용 Dual 모델과 상품 복원용 Canny 모델을 "
                        "한 번만 로드하고 있습니다. "
                        "준비가 끝나면 작업을 자동으로 시작합니다."
                    )
                elif submission.stage == "running_gpu":
                    st.info("공유 GPU Worker가 이미지를 생성하고 있습니다.")
                else:
                    st.info(
                        "GPU 사용을 기다리고 있습니다. "
                        f"등록 시점에 앞에 {submission.queued_ahead}개의 "
                        "작업이 있었습니다."
                    )

                time.sleep(1)
                st.rerun()

            image_result = submission.future.result()
            st.session_state.background_image_path = str(
                image_result.identity_restored_image
            )
            st.session_state.identity_image_path = str(
                image_result.identity_restored_image
            )
            st.session_state.generated_prompt_json = str(
                image_result.prompt_json
            )
            st.session_state.prompt_json_path = str(
                image_result.info_path
            )
            st.session_state.pop("pipeline_submission", None)
            st.rerun()
        except Exception as error:
            st.session_state.pop("pipeline_submission", None)
            traceback.print_exc()
            st.error(f"배경 이미지 생성 중 오류가 발생했습니다: {error}")
            if st.button("⬅ 레이아웃 설정으로 돌아가기"):
                st.session_state.step = 2
                st.rerun()
            st.stop()

    render_step_header(3, "광고 문구 작성 및 최종 완성", "문구 톤과 길이를 선택하면 광고가 완성됩니다.")
    
    # 카피 합성이 완료되었는지 확인하는 상태값
    is_copy_completed = st.session_state.get("is_copy_completed", False)
    
    # ---------------------------------------------------------
    # [A] 카피 합성이 끝나지 않았을 때 -> 레이아웃 (좌: 이미지 / 우: 설정)
    # ---------------------------------------------------------
    if not is_copy_completed:
        # 1. 화면을 좌우 두 칸(1:1 비율)으로 나눕니다.
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.html('<p style="color: #E26848; font-weight: bold; font-size: 14px; margin-bottom: 5px;">GENERATED IMAGE</p>')
            st.html('<h3 style="margin-top: 0;">합성할 이미지 미리보기</h3>')
            st.caption("피사체와 여백을 분석해 문구 배치와 색상을 설계합니다.")
            
            # 👇 STEP 2에서 완성된 "배경 합성 이미지"의 경로(identity_image_path)를 불러옵니다.
            #bg_synthesized_img = st.session_state.get("identity_image_path")
            # 👇 [여기 수정] 1단계에서 완성된 배경 이미지를 가져와 화면에 띄웁니다.
            bg_image_path = st.session_state.get("background_image_path")
            
            # 1. 배경 합성 이미지가 정상적으로 세션에 있을 경우 출력
            if bg_image_path:
                try:
                    st.image(bg_image_path, use_container_width=True)
                except Exception as e:
                    st.error(f"이미지를 화면에 그릴 수 없습니다: {e}")
                    
            # 2. 만약 배경 합성 이미지를 찾지 못한 경우 명확하게 에러 표시
            else:
                st.error("⚠️ STEP 2의 배경 합성 이미지를 찾을 수 없습니다.")
                st.info("💡 팁: STEP 2 코드에서 합성된 결과물이 `st.session_state.identity_image_path = (결과이미지)` 형태로 잘 저장되고 있는지 확인해 주세요!")

        with col2:
            st.html('<p style="color: #E26848; font-weight: bold; font-size: 14px; margin-bottom: 5px;">COPY DIRECTION</p>')
            st.html('<h3 style="margin-top: 0;">광고 문구 생성 설정</h3>')
            st.caption("원하는 인상과 문구 분량을 정해주세요. 생성된 문구는 이미지에 맞춘 레이아웃과 함께 최종 합성됩니다.")
            
            # 폼을 제외하고 일반 UI로 배치하여 하단 버튼과 자유롭게 연동합니다.
            st.html('<p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #31333F; margin-top: 15px;">문구 톤</p>')
            copy_tone = st.radio(
                "문구 톤",
                options=["💡 감성적", "👔 신뢰감", "✨ 트렌디", "😄 유머러스", "🎯 직관적"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.html('<p style="font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #31333F; margin-top: 15px;">글 길이</p>')
            copy_length = st.selectbox("글 길이", ["짧게", "보통", "길게"], label_visibility="collapsed")
            
            # 시각적 디테일: 합성 과정 안내 박스 추가
            st.html("""
            <div style="background-color: #F8F9FA; padding: 15px; border-radius: 8px; margin-top: 20px; border: 1px solid #E9ECEF;">
                <p style="font-size: 13px; font-weight: bold; color: #E26848; margin-bottom: 5px;">합성 과정</p>
                <p style="font-size: 13px; color: #555; margin: 0;">문구 작성 ➔ 이미지 여백 분석 ➔ 타이포그래피와 컬러 설계 ➔ 완성 광고 검토</p>
            </div>
            """)

        st.html("<br><br>")  # 여백 추가
        
        # 2. 하단 액션 버튼 영역 (버튼 두 개를 나란히 배치)
        btn_col1, btn_col2 = st.columns([1, 1])
        
        with btn_col2:
            # 1. 동적 버튼 텍스트 설정 (에러가 나면 버튼 이름이 바뀜)
            btn_label = st.session_state.get("step4_btn_label", "STEP 4: 카피 생성 및 합성 시작 ➔")
            
            if st.button(btn_label, type="primary", use_container_width=True):
                # 에러 메시지 초기화
                if "step4_error_msg" in st.session_state:
                    del st.session_state["step4_error_msg"]
                    
                with st.spinner("AI가 카피를 작성하고 이미지와 합성 중입니다..."):
                    try:
                        import json
                        
                        output_dir = st.session_state.get("job_output_dir")
                        info_json_path = st.session_state.get("prompt_json_path")
                        bg_image_path = st.session_state.get("background_image_path")
                        
                        if not bg_image_path or not info_json_path:
                            raise ValueError("배경 이미지나 설정 파일이 유실되었습니다. STEP 2부터 다시 진행해주세요.")

                        # 1. STEP 3에서 설정한 문구 톤/길이를 기존 JSON 파일에 추가 업데이트
                        with open(info_json_path, "r", encoding="utf-8") as f:
                            info_data = json.load(f)
                            
                        # copy_tone, copy_length 변수는 STEP 3 화면에서 선택한 변수명과 동일해야 합니다.
                        info_data["tone"] = copy_tone
                        info_data["length"] = copy_length
                        
                        with open(info_json_path, "w", encoding="utf-8") as f:
                            json.dump(info_data, f, ensure_ascii=False, indent=4)
                            
                        # 2. 🚨 [추가된 핵심 로직] 2단계: 카피 작성 & 글씨 레이아웃 합성 단독 실행
                        from adcg.pipeline import run_copy_layout_pipeline
                        
                        # 🚨 [수정됨] 'image_path' 대신 파이프라인이 원하는 'identity_image'라는 이름으로 줍니다!
                        # 1단계에서 만들어진 prompt_json 도 함께 넘겨줍니다.
                        result = run_copy_layout_pipeline(
                            identity_image=bg_image_path,
                            info_path=info_json_path,
                            prompt_json=st.session_state.get("generated_prompt_json", info_json_path),
                            output_dir=output_dir
                        )

                        # 최종 결과물 저장
                        st.session_state.final_image_path = str(result.final_image)

                        st.session_state.step4_btn_label = "STEP 4: 카피 생성 및 최종 합성 ➔"
                        st.session_state.is_copy_completed = True
                        st.session_state.step = 4
                        st.rerun()

                    except Exception as e:
                        # 🚨 [핵심] 에러 발생 시 버튼 텍스트를 오류 상태로 바꾸고 리런(새로고침)
                        st.session_state.step4_btn_label = "❌ 오류 발생! (다시 시도)"
                        st.session_state.step4_error_msg = str(e)
                        st.rerun()
            
            # 리런 후 버튼 바로 아래에 상세 에러 메시지 출력
            if st.session_state.get("step4_error_msg"):
                st.error(f"⚠️ {st.session_state.step4_error_msg}")

    # ---------------------------------------------------------
    # [B] 카피 합성이 완료되었을 때 -> 결과 표출 화면
    # ---------------------------------------------------------
    else:
        st.success("🎉 광고 제작이 완료되었습니다!")
        
        # ... (이곳에 완성된 최종 이미지 렌더링 코드 추가) ...
        # 저장된 최종 이미지나 원본 이미지를 화면에 띄웁니다.
        final_image_path = st.session_state.get("final_image_path")
        if final_image_path and Path(final_image_path).is_file():
            st.image(
                final_image_path,
                width="stretch",
                caption="최종 완성된 광고 이미지",
            )
        elif "uploaded_file" in st.session_state and st.session_state.uploaded_file:
            st.image(
                st.session_state.uploaded_file,
                width="stretch",
                caption="광고 이미지 미리보기",
            )
        else:
            st.warning("표시할 이미지가 없습니다.")
        
        st.html("<br>")
        # 하단 돌아가기 버튼
        if st.button("새로운 광고 만들기 🔄", width="stretch"):
            st.session_state.step = 1
            st.session_state.is_copy_completed = False
            st.rerun()


# ---------------------------------------------------------
# [STEP 04] 최종 카피 합성 결과
# ---------------------------------------------------------
if st.session_state.step == 4:
    render_step_header(
        4,
        "완성된 광고 시안",
        "생성된 광고 이미지를 확인하고 다운로드할 수 있습니다.",
    )

    final_image_path = st.session_state.get("final_image_path")
    final_image = Path(final_image_path) if final_image_path else None

    if final_image is not None and final_image.is_file():
        preview_col, details_col = st.columns(
            [0.58, 0.42],
            gap="large",
            vertical_alignment="top",
        )

        with preview_col, st.container(key="final_preview_panel"):
            st.image(
                str(final_image),
                width="stretch",
                caption="최종 완성된 광고 이미지",
            )

        with details_col, st.container(key="final_details_panel"):
            st.success("🎉 광고 제작이 완료되었습니다!")
            st.caption(f"저장 경로: {final_image.resolve()}")

            with final_image.open("rb") as image_file:
                st.download_button(
                    "광고 이미지 다운로드 📥",
                    data=image_file,
                    file_name=final_image.name,
                    mime="image/png",
                    width="stretch",
                )
    else:
        st.error(
            "최종 이미지를 찾을 수 없습니다. "
            f"저장된 경로: {final_image_path or '없음'}"
        )

    st.html("<br>")
    retry_col, restart_col = st.columns(2)

    with retry_col:
        if st.button("✏️ 카피 다시 만들기", width="stretch"):
            st.session_state.step = 3
            st.session_state.is_copy_completed = False
            st.session_state.pop("final_image_path", None)
            st.rerun()

    with restart_col:
        if st.button("새로운 광고 만들기 🔄", width="stretch"):
            st.session_state.step = 1
            st.session_state.is_copy_completed = False
            st.session_state.pop("final_image_path", None)
            st.rerun()
