"""Streamlit demo for local JEV.

Two modes:
  * Demo mode      -- no server needed. A local keyword heuristic fills in a
                      plausible distribution so you can try the UI without a
                      GPU. Results are clearly marked as mock, not model output.
  * Live mode      -- calls a running SGLang server through jev.JevEngine
                      (/v1/score for scoring, /v1/chat/completions for the
                      generation baseline).

Run:  streamlit run app.py
The server URL and model default to JEV_BASE_URL / JEV_MODEL env vars.
"""

from __future__ import annotations

import math
import os
import re
import time

import streamlit as st

# jev/ lives next to this file; sys.path[0] is this dir under `streamlit run`.
from jev import JevEngine


def _default(key: str, fallback: str) -> str:
    """Config lookup: Streamlit secrets first (Community Cloud), then env."""
    try:
        if key in st.secrets:  # raises/empty if no secrets file — that's fine
            return str(st.secrets[key])
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(key, fallback)

# --- presets (kept inline so the app is self-contained) -------------------
PRESETS = {
    "지원 문의 라우팅": {
        "instruction": "이 문의를 어느 팀으로 보내야 하나?",
        "answers": {
            "billing": "요금, 청구, 환불, 결제 문제",
            "technical": "제품 오류, 버그, 장애 등 기술 문제",
            "account": "로그인, 비밀번호, 계정 접속 문제",
            "other": "위 어디에도 맞지 않는 경우",
        },
        "examples": [
            "같은 구독료가 두 번 청구됐어요.",
            "파일을 올릴 때마다 앱이 계속 꺼져요.",
            "비밀번호를 바꿔도 로그인이 안 돼요.",
        ],
    },
    "지원자 선별": {
        "instruction": "이 지원자를 어떻게 처리할까?",
        "answers": {
            "advance": "핵심 요건을 분명히 충족, 면접으로 진행",
            "review": "애매함, 사람이 검토 필요",
            "reject": "핵심 요건을 분명히 미달",
            "other": "판단할 정보가 부족",
        },
        "examples": [
            "역할: 시니어 파이썬 엔지니어(5년+). 지원자: 파이썬 8년, 백엔드 팀 리드.",
            "역할: 데이터 과학자, 운영 ML. 지원자: ML 연구 강점, 운영 경험 없음.",
        ],
    },
    "경비 검토": {
        "instruction": "이 경비를 어떻게 처리할까?",
        "answers": {
            "approve": "정책 내, 자동 승인",
            "flag": "정책 위반 가능성, 관리자에게 전달",
            "deny": "정책 위반, 거부",
            "other": "설명만으로 분류 불가",
        },
        "examples": [
            "공항에서 호텔까지 택시 12,000원, 영수증 첨부.",
            "사전 승인 없는 450만원 일등석 항공권.",
        ],
    },
    "직접 입력": {
        "instruction": "어느 라벨이 입력과 가장 잘 맞나?",
        "answers": {"yes": "그렇다", "no": "아니다", "unclear": "불분명"},
        "examples": [""],
    },
}


# --- mock scoring (demo mode only) ----------------------------------------
def mock_decide(query: str, answers: dict[str, str]) -> dict:
    """A transparent keyword heuristic that mimics decide()'s output shape.

    NOT a model. It rewards word overlap between the query and each answer's
    meaning, adds a tiny deterministic tie-breaker, and softmaxes. Used only
    to exercise the UI when no SGLang server is available.
    """
    q = set(re.findall(r"\w+", query.lower()))
    logits = []
    for ans, meaning in answers.items():
        words = set(re.findall(r"\w+", f"{ans} {meaning}".lower()))
        overlap = len(q & words)
        jitter = (hash(ans) % 100) / 500.0  # 0..0.2, stable per label
        logits.append(1.0 + overlap * 1.6 + jitter)
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    probs = {ans: e / s for ans, e in zip(answers, exps)}
    ranked = sorted(probs.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    return {
        "choice": max(probs, key=probs.get),
        "probabilities": probs,
        "margin": margin,
    }


def parse_answers(text: str) -> dict[str, str]:
    """Parse 'answer = meaning' lines into a dict (meaning defaults to answer)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            ans, meaning = line.split("=", 1)
            out[ans.strip()] = meaning.strip()
        else:
            out[line] = line
    return out


def render_distribution(probs: dict[str, float]) -> None:
    for ans, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        left, right = st.columns([3, 1])
        with left:
            st.progress(min(max(p, 0.0), 1.0), text=ans)
        with right:
            st.markdown(f"**{p:.3f}**")


# --- page -----------------------------------------------------------------
st.set_page_config(page_title="JEV 로컬 결정 엔진", page_icon="⚖️", layout="centered")

with st.sidebar:
    st.header("설정")
    mode = st.radio(
        "실행 모드",
        ["데모 모드 (서버 없이)", "실서버 (SGLang)"],
        help="데모 모드는 모델 없이 화면만 확인합니다. 결과는 모의값입니다.",
    )
    live = mode.startswith("실서버")
    if live:
        base_url = st.text_input(
            "서버 주소", _default("JEV_BASE_URL", "http://127.0.0.1:30000")
        )
        model = st.text_input(
            "모델", _default("JEV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        )
        compare = st.checkbox("생성 방식과 속도 비교", value=False)
    else:
        st.info("모델을 호출하지 않습니다. 확률은 키워드 기반 모의값입니다.")
        compare = False

st.title("JEV · 로컬 결정 엔진")
st.caption(
    "질문과 정해진 후보를 주면, 새 문장을 만들지 않고 후보별 확률을 한 번에 돌려줍니다."
)

preset_name = st.selectbox("예시 유형", list(PRESETS.keys()))
preset = PRESETS[preset_name]

example = st.selectbox("예시 입력", preset["examples"]) if preset["examples"][0] else ""
query = st.text_area("입력(질문)", value=example, height=90, placeholder="여기에 문의나 질문을 입력하세요.")

answers_text = st.text_area(
    "후보 (한 줄에 하나, `라벨 = 설명`)",
    value="\n".join(f"{a} = {m}" for a, m in preset["answers"].items()),
    height=140,
)
instruction = st.text_input("지시문(선택)", value=preset["instruction"])

if st.button("결정하기", type="primary", use_container_width=True):
    answers = parse_answers(answers_text)
    if not query.strip():
        st.warning("입력(질문)을 채워 주세요.")
    elif len(answers) < 2:
        st.warning("후보를 두 개 이상 입력해 주세요.")
    else:
        try:
            if live:
                engine = JevEngine(model=model, base_url=base_url)
                t0 = time.perf_counter()
                r = engine.decide(query, answers, instruction=instruction)
                result = {
                    "choice": r.choice,
                    "probabilities": r.probabilities,
                    "margin": r.margin,
                    "latency_ms": r.latency_ms,
                }
            else:
                t0 = time.perf_counter()
                result = mock_decide(query, answers)
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0

            st.subheader("결정")
            st.success(f"선택: **{result['choice']}**")
            c1, c2 = st.columns(2)
            c1.metric("상위-차상위 차이(margin)", f"{result['margin']:.3f}")
            c2.metric("지연 시간", f"{result['latency_ms']:.1f} ms")

            st.subheader("확률 분포")
            render_distribution(result["probabilities"])

            if not live:
                st.caption("모의 결과입니다. 실제 모델 출력이 아닙니다.")

            # 라우팅 정책은 애플리케이션 코드에 둔다 (모델이 아니라)
            top = result["probabilities"][result["choice"]]
            if top >= 0.70 and result["margin"] >= 0.20:
                st.info(f"자동 처리 대상 (확실): {result['choice']}")
            else:
                st.info(f"사람 검토로 보냄 (top={top:.2f}, margin={result['margin']:.2f})")

            if live and compare:
                st.divider()
                st.subheader("생성 방식과 비교")
                g = engine.generate_response(query, answers, instruction=instruction, max_tokens=32)
                g1, g2 = st.columns(2)
                g1.metric("점수 방식", f"{result['latency_ms']:.1f} ms", result["choice"])
                g2.metric("생성 방식", f"{g.latency_ms:.1f} ms", g.choice or "파싱 실패")
                if result["latency_ms"] > 0:
                    st.caption(f"점수 방식이 약 {g.latency_ms / result['latency_ms']:.1f}배 빠름")
                with st.expander("생성 방식 원문 보기"):
                    st.code(g.text or "(빈 응답)")
        except Exception as exc:  # noqa: BLE001 -- surface any server/parse error in the UI
            st.error(f"요청 실패: {exc}")
            if live:
                st.caption("서버 주소와 SGLang 실행 여부를 확인하세요.")

st.divider()
st.caption(
    "점수 방식은 후보를 미리 정할 수 있을 때만 씁니다. "
    "확률은 후보들 안에서의 몫이며, 모델의 정답률과는 다릅니다."
)
