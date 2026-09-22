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
import statistics
import time
from collections import defaultdict

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

# --- labeled cases for the accuracy check (reuse the preset answer sets) ---
_SUPPORT = PRESETS["지원 문의 라우팅"]["answers"]
_SCREEN = PRESETS["지원자 선별"]["answers"]
_EXPENSE = PRESETS["경비 검토"]["answers"]


def _cases(domain: str, answers: dict, pairs: list[tuple[str, str]]) -> list[dict]:
    return [{"domain": domain, "answers": answers, "query": q, "label": lbl}
            for q, lbl in pairs]


EVAL_CASES = (
    _cases("지원 문의", _SUPPORT, [
        ("같은 구독료가 두 번 청구됐어요.", "billing"),
        ("카드에 중복 결제된 금액을 환불해 주세요.", "billing"),
        ("취소했는데도 요금이 청구됐어요.", "billing"),
        ("파일을 올릴 때마다 앱이 계속 꺼져요.", "technical"),
        ("대시보드를 열면 500 오류가 납니다.", "technical"),
        ("내보내기 버튼을 눌러도 아무 반응이 없어요.", "technical"),
        ("비밀번호를 바꿔도 로그인이 안 돼요.", "account"),
        ("문자로 오는 인증 코드가 오지 않아요.", "account"),
        ("계정 이메일 주소를 바꾸고 싶어요.", "account"),
        ("명절 선물용 상품권도 파나요?", "other"),
    ])
    + _cases("지원자 선별", _SCREEN, [
        ("역할: 시니어 파이썬(5년+). 지원자: 파이썬 8년, 백엔드 팀 리드.", "advance"),
        ("역할: 프런트엔드(React). 지원자: React 6년, 대형 앱 3개 출시.", "advance"),
        ("역할: iOS 개발. 지원자: Swift 4년, 사용자 100만 앱 출시.", "advance"),
        ("역할: 시니어 파이썬(5년+). 지원자: 파이썬 1년, 부트캠프 수료.", "reject"),
        ("역할: 데브옵스(쿠버네티스). 지원자: 윈도우 데스크톱 지원 경력만 있음.", "reject"),
        ("역할: 데이터 과학자, 운영 ML. 지원자: ML 연구 강점, 운영 경험 없음.", "review"),
        ("역할: 제품 분석가(SQL). 지원자: SQL 강함, 도메인 적합성 불명확.", "review"),
        ("역할: 보안 엔지니어. 지원자: 이름 외에 이력서가 비어 있음.", "other"),
    ])
    + _cases("경비 검토", _EXPENSE, [
        ("공항에서 호텔까지 택시 12,000원, 영수증 첨부.", "approve"),
        ("팀 점심 4명 45,000원, 영수증 첨부.", "approve"),
        ("주차비 9,000원, 영수증 있음.", "approve"),
        ("사전 승인 없는 450만원 일등석 항공권.", "flag"),
        ("하루 숙박 호텔 30만원, 도시 상한은 25만원.", "flag"),
        ("영수증과 설명이 없는 '기타' 200만원.", "flag"),
        ("회사 카드로 결제한 30만원 개인 스파.", "deny"),
        ("업무 참석자 없는 주류만 6만원.", "deny"),
    ])
)


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
st.header("정확도 점검")
st.caption(
    "라벨된 예시를 한 번에 돌려 점수 방식의 정확도와 속도를 봅니다. "
    "정확도는 모델 크기에 좌우됩니다(작은 모델일수록 낮음)."
)
n_eval = st.slider("평가할 예시 수", 5, len(EVAL_CASES), len(EVAL_CASES))

if st.button("정확도 평가 실행", use_container_width=True):
    cases = EVAL_CASES[:n_eval]
    engine = JevEngine(model=model, base_url=base_url) if live else None
    prog = st.progress(0.0, text="평가 중...")
    rows, latencies = [], []
    dom_total, dom_ok = defaultdict(int), defaultdict(int)
    correct = 0
    try:
        for i, c in enumerate(cases):
            if live:
                r = engine.decide(c["query"], c["answers"])
                choice, ms = r.choice, r.latency_ms
            else:
                t0 = time.perf_counter()
                choice = mock_decide(c["query"], c["answers"])["choice"]
                ms = (time.perf_counter() - t0) * 1000.0
            ok = choice == c["label"]
            correct += ok
            latencies.append(ms)
            dom_total[c["domain"]] += 1
            dom_ok[c["domain"]] += int(ok)
            rows.append({
                "도메인": c["domain"],
                "입력": c["query"][:26],
                "정답": c["label"],
                "예측": choice,
                "맞음": "O" if ok else "X",
                "ms": round(ms, 1),
            })
            prog.progress((i + 1) / len(cases), text=f"평가 중... {i + 1}/{len(cases)}")
    except Exception as exc:  # noqa: BLE001 -- surface server/parse errors in the UI
        prog.empty()
        st.error(f"평가 실패: {exc}")
        if live:
            st.caption("서버 주소와 SGLang 실행 여부를 확인하세요.")
    else:
        prog.empty()
        m1, m2, m3 = st.columns(3)
        m1.metric("정확도", f"{correct / len(cases):.0%}", f"{correct}/{len(cases)}")
        m2.metric("평균 지연", f"{statistics.mean(latencies):.1f} ms")
        m3.metric("중앙값 지연", f"{statistics.median(latencies):.1f} ms")

        st.caption("도메인별 정확도")
        render_distribution({d: dom_ok[d] / dom_total[d] for d in dom_total})

        with st.expander("케이스별 결과 보기"):
            st.dataframe(rows, use_container_width=True, hide_index=True)

        if not live:
            st.caption(
                "데모 모드: 예측이 키워드 기반 모의값이라 정확도가 낮습니다. "
                "실서버 모드에서 실제 정확도를 확인하세요."
            )

st.divider()
st.caption(
    "점수 방식은 후보를 미리 정할 수 있을 때만 씁니다. "
    "확률은 후보들 안에서의 몫이며, 모델의 정답률과는 다릅니다."
)
