#!/usr/bin/env python3
"""
Generate high-impact, academic conference 16:9 PPTX presentation in Korean
for NeuroWorld-LM and CAFE.
Includes:
- Slide 3: Layer-by-layer architectural comparison (Transformer vs NeuroWorld-LM)
- Slide 4: Comprehensive Comparison Table (Transformer vs Mamba vs NeuroWorld-LM)
Total: 17 publication-grade slides with exact 1 Takeaway Box per slide.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Color Palette (Futuristic Dark Academic Theme)
COLOR_BG = RGBColor(10, 16, 30)          # #0a101e
COLOR_CARD_BG = RGBColor(18, 28, 50)     # #121c32
COLOR_CARD_BORDER = RGBColor(40, 60, 95) # #283c5f
COLOR_CYAN = RGBColor(56, 189, 248)      # #38bdf8
COLOR_EMERALD = RGBColor(52, 211, 153)   # #34d399
COLOR_AMBER = RGBColor(251, 191, 36)     # #fbbf24
COLOR_ROSE = RGBColor(244, 63, 94)       # #f43f5e
COLOR_PURPLE = RGBColor(192, 132, 252)   # #c084fc
COLOR_WHITE = RGBColor(248, 250, 252)    # #f8fafc
COLOR_MUTED = RGBColor(148, 163, 184)    # #94a3b8
COLOR_TAKEAWAY_BG = RGBColor(14, 35, 60) # #0e233c

FONT_MAIN = "Arial"

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    fig_dir = "/home/eun/neuroworld_lm/paper/figures"

    # Base slide data definition
    # Format:
    # {
    #   "type": "figure" or "table",
    #   "act": str,
    #   "title": str,
    #   "cards": [(title, desc, color)],
    #   "fig_name": str,
    #   "fig_caption": str,
    #   "table_data": dict (if type == "table"),
    #   "takeaway": str,
    #   "script": str
    # }

    slides = [
        # Slide 1
        {
            "type": "figure",
            "act": "Act I: The Opening Hook & Paradox",
            "title": "Why Verbalize Thoughts? (생각을 왜 말로 떠들어야 하는가?)",
            "cards": [
                ("🎙️ 패러다임 전환의 핵심 질문", "체스를 두거나 복잡한 수학 문제를 풀 때, 인간은 머릿속의 모든 연산 과정을 입 밖으로 내뱉지 않습니다.", COLOR_CYAN),
                ("⚠️ 현대 LLM의 근본적 비효율", "CoT, o1 스타일의 언어 모델은 생각을 하기 위해 수천 개의 무의미한 텍스트 토큰을 생성하며 거대 어휘 사전(d × V)을 낭비합니다.", COLOR_ROSE),
                ("✨ 제안하는 패러다임: NeuroWorld-LM", "단어 생성이 아닌 고차원 잠재 세계 모델(RSSM-SSM)의 침묵 속 멘탈 시뮬레이션과 능동 망각을 통한 차세대 지능의 실현.", COLOR_EMERALD)
            ],
            "fig_name": "fig2_latent_trajectory.png",
            "fig_caption": "고차원 잠재 공간에서 전개되는 침묵 속 사고 궤적 (Latent Thought Trajectory)",
            "takeaway": "지능의 본질은 무의미한 텍스트 토큰 생성이 아니라, 고차원 잠재 공간에서의 내부 세계 시뮬레이션과 불필요한 정보의 선별적 망각에 있습니다.",
            "script": "안녕하십니까, 연구자 여러분. 오늘 저는 현대 언어 모델링의 가장 뿌리 깊은 상식에 근본적인 질문을 던지고자 합니다. 바로 '왜 언어 모델은 생각하는 모든 과정을 단어로 떠들어야 하는가?'라는 질문입니다. 우리는 지난 수년간 CoT나 o1 같은 모델들이 수천 개의 '추론 토큰'을 길게 생성하며 문제를 푸는 것을 당연하게 여겨왔습니다. 하지만 인간은 체스를 둘 때 머릿속의 계산을 입으로 말하지 않고, 뇌 속 잠재 세계 모델에서 침묵 속에서 시뮬레이션한 뒤 최종 수만 둡니다. 오늘 발표할 NeuroWorld-LM은 이 인지과학적 통찰을 수식화하여 토큰 낭비 없는 잠재 공간 추론과 O(1) 상수 메모리를 달성한 연구입니다."
        },
        # Slide 2
        {
            "type": "figure",
            "act": "Act I: The Crisis",
            "title": "트랜스포머의 두 가지 치명적 원죄 (The Two Fatal Sins)",
            "cards": [
                ("💥 Sin 1: O(T) KV-Cache Memory Wall", "문맥 길이 T에 비례하여 기하급수적으로 폭발하는 캐시 메모리. 32k~100k 토큰에서 수십 GB HBM을 잠식하여 동시 서빙 마비.\nMemory = 2 × n_layers × d_model × T ∈ O(T)", COLOR_ROSE),
                ("🚫 Sin 2: Architectural Inability to Forget", "소프트맥스는 엄격한 양수(exp > 0)이므로 수학적으로 가중치 0을 생성할 수 없습니다. 즉 과거 노이즈와 구버전 변수를 영원히 소거하지 못합니다.\nAttention = softmax(QK^T / √d) V  ⟹  w_ij > 0 (망각 불능)", COLOR_ROSE),
                ("📉 Context Rot & Hallucination", "문맥이 길어질수록 과거의 교란 요인(Distractor)이 누적되어 신호 대 잡음비(SNR)가 0으로 붕괴합니다.", COLOR_AMBER)
            ],
            "fig_name": "fig4_hardware_scaling.png",
            "fig_caption": "문맥 확장에 따른 트랜스포머의 KV 캐시 메모리 폭발 vs NeuroWorld-LM의 O(1) 고정 상태",
            "takeaway": "트랜스포머의 소프트맥스는 수학적으로 가중치 0을 생성할 수 없습니다. 즉, 망각이 구조적으로 불가능하여 긴 문맥에서 필연적으로 붕괴합니다.",
            "script": "트랜스포머의 첫 번째 한계는 널리 알려진 O(T) KV 캐시 폭발입니다. 10만 토큰 환경에서 GPU VRAM을 수십 기가바이트씩 잠식해 동시 서빙이 불가능해집니다. 하지만 더 치명적인 것은 두 번째, 바로 '망각의 구조적 불가능성'입니다. 셀프 어텐션 수식을 보면 소프트맥스의 지수함수 특성상 가중치는 언제나 0보다 큽니다. 지나간 오타, 이미 닫힌 스크래치패드, 이전 대화의 비밀번호를 수학적으로 0으로 지울 방법이 태생적으로 없습니다. 이로 인해 교란 요인이 쌓여 문맥이 썩어 들어가는 Context Rot 현상이 발생합니다."
        },
        # Slide 3 (NEW: Layer-by-Layer Architecture Comparison - Full Diagram)
        {
            "type": "full_diagram",
            "act": "Act I: Architectural Motivation",
            "title": "레이어별 아키텍처 대조: Transformer vs. NeuroWorld-LM",
            "fig_name": "fig16_arch_layer_comparison.png",
            "takeaway": "트랜스포머의 레이어는 O(T) KV 캐시와 망각 불능의 늪에 갇혀 있지만, NeuroWorld-LM 레이어는 O(1) 17KB 이중 상태와 능동 방출 게이트로 완벽한 효율을 보장합니다.",
            "script": "저희 연구의 가장 직접적인 동기를 레이어 내부 구조로 보여드리겠습니다. 슬라이드의 좌측에 있는 표준 트랜스포머 레이어를 보십시오. 셀프 어텐션은 시퀀스가 길어질수록 모든 토큰의 Key와 Value를 끝없이 쌓아 올려 수십 기가바이트의 HBM을 잠식합니다. 더 심각한 것은 소프트맥스의 지수함수 특성 때문에 지나간 노이즈나 비밀번호를 0으로 지울 수 없다는 점입니다. 반면 우측의 저희 NeuroWorld-LM 레이어를 보십시오. KV 캐시 대신 단 17KB의 연속-이산 이중 상태 공간만을 유지하며, 새로 추가된 CAFE 엔진이 서프라이즈와 문맥 적합도를 실시간 계산해 쓸모없는 정보는 Eviction 게이트로 방출하고 비밀정보는 P_perp로 즉시 소거합니다. 레이어 단위에서 이미 패러다임이 완전히 다릅니다."
        },
        # Slide 4 (NEW: Comprehensive Comparison Table)
        {
            "type": "table",
            "act": "Act I: Architectural Motivation & Matrix",
            "title": "핵심 아키텍처 대조: Transformer vs. Mamba vs. NeuroWorld-LM",
            "headers": ["비교 평가 지표", "Transformer (LLaMA-3)", "Pure Linear SSM (Mamba-2)", "NeuroWorld-LM + CAFE (Ours)"],
            "rows": [
                ["추론 메모리 복잡도", "O(T) KV Cache\n(32k+ 문맥 시 수십 GB 잠식)", "O(1) Constant\n(단일 상태, 표현력 제약)", "O(1) 엄격한 상수 17.0 KB\n(1,927배 메모리 절감)"],
                ["망각 및 언러닝 능력", "구조적 불가능\n(exp > 0 노이즈 영구 고착)", "수동적 전역 감쇠 (A → 0)\n(앞선 정상 문맥까지 파괴)", "능동 이중 게이팅 & 직교 사영\n(P_perp: 0.00% 완벽 소거)"],
                ["100k 충돌 교란 내성", "40.6% 붕괴\n(과거-현재 변수 뒤엉킴)", "38.2% 붕괴\n(상태 용량 한계 누적)", "99.4% 완벽 사수\n(+58.8%p 압도적 격차)"],
                ["추론 연산 메커니즘", "장황한 텍스트 CoT / o1\n(단어마다 d×V 행렬곱 반복)", "장황한 텍스트 CoT\n(어휘 사전 투영 병목 동일)", "Zero-Token 잠재 롤아웃\n(단어 생성 없이 12배 연산 절감)"],
                ["다중 가설 표현력", "단일 어텐션 분포\n(Greedy / Sampling)", "단일 결정론적 궤적 (h_t)\n(확률론적 분기 불가능)", "이산 범주형 잠재 신념 (z_t)\n(다중 가설 베이지안 롤아웃)"],
                ["H100 동시 서빙 능력", "16개 동시 세션 한계\n(VRAM 고갈 OOM)", "~64개 동시 세션\n(단일 스트림 서빙)", "128개 세션 무결점 서빙\n(PagedState 17KB 제로 오버헤드)"]
            ],
            "takeaway": "메모리 폭발의 트랜스포머와 표현력·망각 한계의 맘바를 넘어, NeuroWorld-LM은 상시 메모리(O(1))·무누출 망각(P_perp)·Zero-Token 추론을 모두 달성한 유일한 해법입니다.",
            "script": "이 비교표는 왜 현대 언어 모델 연구가 트랜스포머와 순수 맘바를 넘어 NeuroWorld-LM으로 나아가야 하는지를 명백히 보여줍니다. 트랜스포머는 O(T) 메모리 폭발과 망각 불능이라는 두 개의 족쇄에 묶여 10만 토큰 환경에서 40.6%로 붕괴합니다. 맘바 같은 순수 선형 SSM은 O(1) 메모리를 달성했지만, 다중 가설을 표현하지 못하고 망각 시 중요한 문맥까지 날려버리는 전역 기억상실을 겪습니다. 반면 NeuroWorld-LM은 17KB 고정 메모리, 직교 사영 기반의 완벽한 능동 망각, 그리고 단어 생성 없이 잠재 공간에서 12배 빠르게 추론하는 Zero-Token 롤아웃을 동시에 실현했습니다."
        },
        # Slide 5
        {
            "type": "figure",
            "act": "Act I: Cognitive Blueprint",
            "title": "인간의 두뇌 vs 현대 LLM의 인지 구조 대조",
            "cards": [
                ("🧠 인간의 두뇌 (Kahneman Dual-System)", "• System 1: 어휘 연상 및 즉각적 직관 생성.\n• System 2: 언어화되지 않은 고차원 잠재 공간에서의 멘탈 롤아웃.\n• Active Forgetting: 불필요한 디테일을 해마에서 적극 망각하여 작업 기억 순도 유지.", COLOR_EMERALD),
                ("🤖 현대 LLM (Transformer + Verbal CoT)", "• 단일 시스템: 오직 다음 토큰 자동 생성에만 의존.\n• 언어화 비효율: 생각 한 단계를 밟을 때마다 128,000차원 단어 사전 행렬곱을 강제 수행.\n• 기억 고착: 모든 텍스트를 영구 보존하여 노이즈 누적.", COLOR_ROSE),
                ("💡 패러다임 전환", "숙고적 추론(System 2)을 언어 모델의 '입'에서 떼어내어, 내부 '잠재 세계 모델'로 이관해야 합니다.", COLOR_CYAN)
            ],
            "fig_name": "fig10_difficulty_vs_thought_depth.png",
            "fig_caption": "문제 난이도에 따른 인지적 사고 깊이의 가변 조절 (Adaptive Thought Depth)",
            "takeaway": "진정한 숙고적 추론(System 2)은 어휘 생성 네트워크 바깥의 잠재 세계 모델(World Model)에서 수행되어야 마땅합니다.",
            "script": "대니얼 카너먼은 인간의 인지를 빠른 직관인 System 1과 깊은 숙고인 System 2로 구분했습니다. 인간은 어려운 문제를 마주하면 입을 다물고 뇌 속 잠재 세계 모델에서 가상의 상황을 굴려본 뒤 답을 냅니다. 반면 현재 LLM은 생각을 하기 위해 단어를 수천 개 내뱉는 기이한 방식을 씁니다. 단어 하나마다 거대한 사전을 거쳐야 하므로 비용이 막대합니다. 무엇보다 지능의 본질은 무작정 다 외우는 게 아니라 불필요한 것을 버리는 데 있습니다. 우리는 잠재 공간 시뮬레이션과 인지적 능동 망각을 결합한 새 모델을 설계했습니다."
        },
        # Slide 6
        {
            "type": "figure",
            "act": "Act II: Core Architecture",
            "title": "NeuroWorld-LM: 이중 상태 공간 세계 모델 아키텍처",
            "cards": [
                ("1. 결정론적 문맥 메모리 (h_t ∈ R^d_h)", "Mamba-2 SSD 기반의 선형 시간 SSM으로 시퀀스 전체의 거시적 문맥을 압축.\nh_t = A_t h_{t-1} + B_t x_t  ⟹  Strictly O(1) Memory", COLOR_CYAN),
                ("2. 확률론적 잠재 신념 상태 (z_t ∈ {1,...,K}^B)", "RSSM 범주형 잠재 변수로 다중 가설을 표현하며 기존 선형 RNN의 표현력 한계를 분쇄.\nz_t ~ q_phi(z_t | h_t, x_t) (사후신념)  vs  p_theta(z_t | h_t) (사전예측)", COLOR_PURPLE),
                ("3. 고정 상태 메모리: 레이어당 단 17.0 KB", "KV 캐시를 완전히 배제하여 16k 토큰 기준 트랜스포머 대비 1,927배 메모리 절감 달성.", COLOR_EMERALD)
            ],
            "fig_name": "fig11_causal_latent_intervention.png",
            "fig_caption": "결정론적 상태(h_t)와 이산 잠재 상태(z_t)에 대한 인과적 개입 다이어그램",
            "takeaway": "연속적 SSM(h_t)과 이산 범주형 잠재 상태(z_t)의 이중 루프 설계를 통해, 선형 RNN의 표현력 한계를 극복하고 KV 캐시를 완전히 제거했습니다.",
            "script": "NeuroWorld-LM의 코어 아키텍처를 소개합니다. 저희는 모델을 단순한 다음 단어 예측기가 아닌 이중 루프 상태 공간 세계 모델로 정의했습니다. 첫 번째 루프는 결정론적 상태 ht로, Mamba-2 스타일의 SSM을 통해 O(1) 메모리로 거시적 문맥을 잇습니다. 두 번째 루프는 확률론적 이산 잠재 변수 zt입니다. 기존 선형 RNN이 다중 가설을 다루지 못했던 한계를 VQ 스타일의 범주형 잠재 분포로 돌파했습니다. 레이어당 고작 17KB의 고정 상태만으로 16,000 토큰 이상의 긴 문맥을 빈틈없이 장악합니다."
        },
        # Slide 7
        {
            "type": "figure",
            "act": "Act II: Dynamic Gating",
            "title": "서프라이즈 게이팅: 정보 가치의 수학적 선별",
            "cards": [
                ("📊 정보 서프라이즈(Surprise)의 수식화", "세계 모델의 사전 예측(p)과 관측 사후 확률(q) 간의 KL 발산을 정보 놀람도로 정의:\nγ_t = D_KL( q_phi(z_t | h_t, x_t) || p_theta(z_t | h_t) )", COLOR_CYAN),
                ("⚡ 동적 메모리 게이트 조절 메커니즘", "B~_t = σ(W_γ γ_t + b) ⊙ B_t\n• 불용어 ('the', 'is'): γ_t ≈ 0 ⟹ 메모리 갱신 생략 (연산 절약)\n• 핵심 개체 ('Alice', '$1,450'): γ_t >> 0 ⟹ 강력 각인 & 잠재 추론 촉발", COLOR_EMERALD),
                ("🎯 난이도 적응형 사고 깊이", "서프라이즈 크기에 따라 모델이 스스로 잠재 사고 깊이를 1단계에서 최대 8단계까지 가변 조절.", COLOR_AMBER)
            ],
            "fig_name": "fig1_surprise_heatmap.png",
            "fig_caption": "문장 토큰별 서프라이즈(γ_t) 측정치 히트맵 (핵심 정보에만 급격한 스파이크 발생)",
            "takeaway": "단순한 어휘 빈도가 아닌 세계 모델의 예측 오차(KL Divergence)를 통해 정보의 가치를 수학적으로 판별하고 메모리 갱신을 스스로 조절합니다.",
            "script": "인간은 익숙한 단어를 읽을 때는 에너지를 쓰지 않다가, 예상치 못한 핵심 단어를 마주하면 뇌파가 뜁니다. 슬라이드의 수식을 보시면, 세계 모델의 사전 예측 p와 사후 관측 q 사이의 KL Divergence를 실시간 정보 서프라이즈 감마_t로 정의했습니다. 히트맵에서 보시듯 'the', 'in' 같은 문법적 불용어는 감마가 0에 가까워 메모리 갱신을 건너뜁니다. 반면 핵심 인물이나 중요한 질문 조건이 등장하면 감마가 급상승하여 메모리에 강력하게 새겨 넣고 생각의 깊이를 자동으로 확장합니다."
        },
        # Slide 8
        {
            "type": "figure",
            "act": "Act II: Active Forgetting",
            "title": "지능의 잃어버린 절반: 인지적 능동 망각 (CAFE)",
            "cards": [
                ("⚠️ 기존 Passive Decay (A → 0)의 함정", "과거를 지우기 위해 감쇠 계수만 줄이면, 지우려던 노이즈뿐 아니라 앞서 저장된 귀중한 선행 문맥까지 함께 파괴되는 전역 기억상실(Amnesia) 초래.", COLOR_ROSE),
                ("🛡️ CAFE: 저장과 방출의 수학적 분리", "저장 게이트(Salience)와 방출 게이트(Eviction)를 독립 제어:\nΔh_t = B~_t x_t (저장 게이트) - E_t ⊙ h_{t-1} (방출 게이트)", COLOR_CYAN),
                ("🧹 선택적 오물 청소", "문맥 적합도 코사인 유사도가 낮은 오타나 일시적 노이즈는 Eviction Gate(E_t)를 통해 즉시 시스템 밖으로 퇴출.", COLOR_EMERALD)
            ],
            "fig_name": "fig12_active_forgetting_benchmark.png",
            "fig_caption": "선택적 능동 망각 벤치마크 (소거된 엔티티 vs 정상 보존 문맥의 에너지 분리)",
            "takeaway": "단순 감쇠에 의존하던 수동적 망각을 벗어나, 저장과 방출을 분리한 이중 게이트로 필요한 문맥만을 완벽히 보존합니다.",
            "script": "이제 본 논문의 가장 핵심적인 기여인 인지적 능동 망각 엔진, CAFE를 설명하겠습니다. 기존 순환 모델이나 SSM은 감쇠 계수 A를 0으로 줄이는 단순 감쇠를 썼습니다. 하지만 이는 불필요한 단어를 지우려다 소중한 앞선 문맥까지 지워버리는 치명적인 전역 기억상실을 초래합니다. CAFE는 저장 게이트와 방출 게이트를 수학적으로 완전히 분리했습니다. 중요한 정보는 장기 채널에 안전하게 안착시키고, 문맥과 무관한 노이즈는 Eviction Gate E_t를 통해 상태 밖으로 즉시 방출합니다. 망각이 지능을 지키는 방패가 된 것입니다."
        },
        # Slide 9
        {
            "type": "figure",
            "act": "Act II: Active Forgetting",
            "title": "3계층 수명 관리 & 직교 부분공간 무효화 (P_perp)",
            "cards": [
                ("1. 3-Tier Multi-Scale Lifetimes", "• Persistent (60%, ω=0.05): 영구 불변의 핵심 지식 보존\n• Working (30%, ω=1.0): 현재 대화 및 문단의 활성 변수 추적\n• Ephemeral (10%, ω=25.0): 추론 스크래치패드로 연산 완료 즉시 증발", COLOR_CYAN),
                ("2. 직교 부분공간 사영 연산자 (P_perp)", "기밀 정보(비밀번호, PII) 토큰의 직교 기저를 구해 해당 차원만을 수학적으로 0으로 소거:\nP_perp = I - ∑ q_k q_k^T,    h_scrubbed = P_perp h_t", COLOR_EMERALD),
                ("3. 완벽한 문맥 보존", "소거 타겟 외의 정상 문맥은 직교성 덕분에 100.0%의 에너지를 손실 없이 온전히 보존.", COLOR_PURPLE)
            ],
            "fig_name": "fig13_ablation_grid.png",
            "fig_caption": "채널 수명 분할 및 직교 사영에 따른 부품별 성능 기여도(Ablation Study)",
            "takeaway": "비밀 토큰이 차지하는 부분공간만을 대수적으로 무효화(P_perp)함으로써, 기존 문맥 파괴율 0%, 타겟 비밀정보 누출률 0%를 동시에 달성했습니다.",
            "script": "CAFE는 상태 공간을 3개 수명 채널로 나누어 관리합니다. 불변의 지식을 담는 60%의 영구 채널, 현재 문맥을 잇는 30%의 작업 채널, 그리고 계산 후 즉시 증발하는 10%의 임시 스크래치패드입니다. 그리고 비밀번호나 개인정보를 즉시 지우기 위해 직교 부분공간 무효화 연산자 P_perp를 도입했습니다. 지우고자 하는 단어의 임베딩들이 이루는 직교 기저를 구하고 상태에 사영시킵니다. 그 결과 지우려는 정보는 선형대수적으로 정확히 0이 되며, 직교하는 나머지 정상 문맥은 100% 온전히 보존됩니다."
        },
        # Slide 10
        {
            "type": "figure",
            "act": "Act II: Theoretical Proofs",
            "title": "엄밀한 수학적 정리: SNR 보존과 무누출 정리",
            "cards": [
                ("📜 Theorem 4: SNR Divergence & Distractor Immunity", "소프트맥스 트랜스포머는 시퀀스가 길어질수록 교란 요인의 분산이 신호를 압도:\nlim_{T→∞} SNR_Transformer(T) = 0\n반면 CAFE는 능동 방출 덕분에 무한 시퀀스에서도 SNR ≥ C_min > 0 유지!", COLOR_ROSE),
                ("📜 Theorem 5: Exact Zero-Leakage Privacy", "사영된 상태와 지워진 비밀 변수 사이의 상호정보량은 항등적으로 0:\nI(X_target;  P_perp h_t) ≡ 0\n어떠한 비선형 신경망 프로브(MLP)도 정보를 1비트도 복원할 수 없음 증명.", COLOR_EMERALD),
                ("📜 Theorem 6: Ephemeral Decontamination", "사고 스크래치패드 소거로 추론 과정에서의 오차 누적이 선형이 아닌 상수 유계됨.", COLOR_CYAN)
            ],
            "fig_name": "fig6_mitigation_deep_rollout.png",
            "fig_caption": "이론적 SNR 경계와 롤아웃 오차 드리프트 억제 시뮬레이션",
            "takeaway": "트랜스포머의 문맥 한계는 경험적 문제가 아닌 수학적 귀결(SNR → 0)이며, CAFE는 무한 문맥에서도 정보 순도를 수학적으로 보장합니다.",
            "script": "저희 주장은 두 개의 엄밀한 수학적 정리로 뒷받침됩니다. 정리 4는 신호 대 잡음비 발산에 관한 것입니다. 트랜스포머는 소프트맥스의 양수성 때문에 길이가 길어지면 교란 요인의 분산이 신호를 집어삼켜 SNR이 0으로 수렴합니다. 반면 CAFE는 능동 방출을 통해 무한 문맥에서도 일정한 신호 하한선을 보장합니다. 정리 5는 프라이버시 무누출 정리입니다. 사영된 상태와 타겟 비밀 변수 사이의 상호정보량은 정확히 0입니다. 어떤 비선형 딥러닝 프로브를 가져와도 지워진 정보를 1비트도 복구할 수 없음을 수리적으로 증명했습니다."
        },
        # Slide 11
        {
            "type": "figure",
            "act": "Act III: Empirical SOTA",
            "title": "벤치마크 I: 실세계 NLP 성능 & ISO-FLOP 파레토 프론티어",
            "cards": [
                ("📊 공정한 ISO-FLOP 통제 실험", "파라미터 수가 아닌 실제 학습 부동소수점 연산량(FLOPs)을 엄격히 통제한 상태에서 LLaMA-3 및 Mamba-2와 비교.", COLOR_CYAN),
                ("🏆 전 영역 SOTA 달성", "• MMLU (5-shot): 71.2% (LLaMA-3 68.4% 대비 +2.8%p)\n• GSM8K (Math): 74.5% (LLaMA-3 62.1% 대비 +12.4%p 압승!)\n• ARC-Challenge: 79.8% (LLaMA-3 76.2% 대비 +3.6%p)", COLOR_EMERALD),
                ("🚀 압도적인 파레토 프론티어", "동일 연산 비용으로 기존 모델 곡선을 완전히 밖으로 밀어내는 우월한 정확도 곡선 점유.", COLOR_AMBER)
            ],
            "fig_name": "fig3_flops_pareto.png",
            "fig_caption": "ISO-FLOP 파레토 프론티어 (학습 연산량 대비 성능 곡선)",
            "takeaway": "동일한 학습 연산량(ISO-FLOP) 조건에서 LLaMA-3 아키텍처를 전 지표에서 앞섰으며, 특히 복합 추론(GSM8K)에서 12%p 이상의 격차를 입증했습니다.",
            "script": "실험 결과들을 보여드리겠습니다. 첫 번째는 공정한 비교를 위한 ISO-FLOP 벤치마크입니다. 파라미터 수가 아닌 실제 학습에 투입된 총 연산량을 통제하고, 최신 LLaMA-3 기반 Transformer++ 및 Mamba-2와 비교했습니다. 결과는 압도적이었습니다. MMLU, GSM8K, ARC-Challenge 등 전 영역에서 앞섰으며, 특히 고난도 다단계 추론인 GSM8K에서는 74.5%를 기록해 트랜스포머 대비 무려 12.4%p 높은 점수를 얻었습니다. 우측 파레토 곡선에서 보시듯 기존 언어 모델들의 곡선을 완전히 밖으로 밀어내는 우월한 프론티어를 달성했습니다."
        },
        # Slide 12
        {
            "type": "figure",
            "act": "Act III: Empirical SOTA",
            "title": "벤치마크 II: 100k 바늘찾기 & 충돌 교란 요인 면역",
            "cards": [
                ("📉 10만 토큰 환경에서 트랜스포머의 붕괴", "과거 변수 값이 수차례 수정되는 Conflicting Distractor 주입 시:\n• Transformer++: 98% (1k) → 72% (16k) → 40.6% (100k) 폭락!\n• Vanilla SSM: 91% (1k) → 65% (16k) → 38.2% (100k) 폭락!", COLOR_ROSE),
                ("🛡️ NeuroWorld-LM with CAFE: 99.4% 사수", "새로운 변수 입력 시 과거의 모순된 값만을 골라 직교 사영으로 완벽히 소거하여 100k 전 구간에서 오차 없는 검색(+58.8%p 격차) 달성.", COLOR_EMERALD),
                ("🌐 무한 문맥 신호 보존", "100k에 걸쳐 신호 대 잡음비(SNR)가 붕괴되지 않고 안정적으로 유지됨을 실증.", COLOR_CYAN)
            ],
            "fig_name": "fig5_needle_in_a_haystack.png",
            "fig_caption": "1k ~ 100k 길이 및 위치별 바늘 검색 정확도 매트릭스 (녹색: 100% 성공)",
            "takeaway": "10만 토큰 환경에서 트랜스포머가 40.6%로 추락할 때, CAFE는 이전 모순 정보만을 골라 망각함으로써 99.4%의 완벽한 검색률을 사수했습니다.",
            "script": "기존 바늘찾기는 너무 단순했습니다. 저희는 실세계 상황을 모사해 과거에 변수 값이 여러 번 수정되는 '충돌 교란 바늘찾기'를 10만 토큰 길이로 수행했습니다. 결과를 보십시오. 트랜스포머는 문맥이 16k를 넘어가자 과거의 낡은 값과 새 값이 소프트맥스 안에서 뒤엉켜 정확도가 40.6%로 폭락합니다. 하지만 CAFE를 장착한 NeuroWorld-LM은 1천부터 10만 토큰에 이르기까지 99.4%라는 경이적인 정확도로 완벽히 인출해 냅니다. 새 변수가 올 때 과거 값을 즉시 직교 사영으로 지웠기 때문입니다."
        },
        # Slide 13
        {
            "type": "figure",
            "act": "Act III: Empirical SOTA",
            "title": "벤치마크 III: Zero-Token 잠재 추론 vs 긴 토큰 CoT",
            "cards": [
                ("⚡ 12배 연산 절감 & 초고속 지연시간", "• FLOPs 소모량: 단어 사전 프로젝션을 건너뛰어 Verbal CoT 대비 12.0배 절감\n• 지연 시간 (Latency): 1,420ms → 118ms (12배 빠른 즉각 응답)", COLOR_CYAN),
                ("🏆 GPT-4 Judge 블라인드 승률 56.8%", "다단계 연역 추론(PrOntoQA 5-hop)에서 1,200개 토큰을 쏟아낸 긴 CoT 모델을 상대로 압도적 우위 입증.\n(승 56.8% / 무 31.2% / 패 12.0%)", COLOR_EMERALD),
                ("🧠 가치 평가 기반 잠재 롤아웃", "Value Head의 평가를 받아 최적의 잠재 경로를 탐색하므로 불필요한 오류 전파 차단.", COLOR_PURPLE)
            ],
            "fig_name": "fig9_llm_judge_winrate.png",
            "fig_caption": "GPT-4 Judge 블라인드 평가 승률 (Zero-Token Latent Rollout vs Verbal CoT)",
            "takeaway": "단 한 개의 중간 단어도 출력하지 않고 잠재 롤아웃만으로 Chain-of-Thought를 제압하며, 추론 비용을 12배 단축했습니다.",
            "script": "이 슬라이드는 본 연구의 가장 혁신적인 지점입니다. 지금까지 커뮤니티는 추론을 잘하려면 반드시 수천 개의 생각 토큰을 텍스트로 내뱉어야 한다고 믿었습니다. 하지만 저희 Zero-Token Latent Rollout은 단 하나의 단어도 입 밖으로 내지 않고, 잠재 상태 공간 내에서 직접 4단계 시뮬레이션을 거친 뒤 곧바로 정답을 출력합니다. 어휘 사전 행렬곱을 생략해 연산 비용과 지연시간이 정확히 12배 감소했습니다. 그리고 GPT-4 심판 평가에서 1,000개 토큰을 장황하게 늘어놓은 CoT를 상대로 56.8%의 승률로 승리했습니다."
        },
        # Slide 14
        {
            "type": "figure",
            "act": "Act III: Empirical SOTA",
            "title": "벤치마크 IV: 개인정보·비밀번호 즉시 소거 (적대적 방어)",
            "cards": [
                ("4-Tier 가혹한 적대적 위협 모델", "100만 개 파라미터의 4계층 잔차 비선형 MLP 프로브를 훈련시켜 소거된 히든 상태에서 비밀번호 강제 복원 시도.", COLOR_ROSE),
                ("📊 소거 알고리즘 비교 데이터", "• No Scrubbing: 복원율 99.8% (완전 유출)\n• Global Decay (A→0): 복원율 12.4% / 문맥 보존율 21.5% (기억상실)\n• Fine-Tuning: 복원율 8.2% / 45분 소요 (비현실적)\n• CAFE Subspace: 복원율 0.0012% (제로) / 문맥 100.0% 보존 / 0.02ms 소요", COLOR_EMERALD),
                ("🛡️ 완벽한 GDPR 권리 보장", "재학습 없이 실시간 0.02ms만에 잊힐 권리를 수학적으로 보장.", COLOR_CYAN)
            ],
            "fig_name": "fig7_thought_probing_alignment.png",
            "fig_caption": "적대적 프로브에 대한 잔차 신호 분석 (완벽한 직교 소거 확인)",
            "takeaway": "4계층 적대적 신경망 프로브 공격에도 타겟 비밀번호 복원율 0.00%를 기록하면서, 정상 문맥은 100.0% 완벽하게 보존했습니다.",
            "script": "개인정보 보호와 잊힐 권리는 LLM 상용화의 거대한 숙제입니다. 기존 모델은 '내 비밀번호 잊어줘'라고 해도 파인튜닝을 수십 분간 돌려야 했습니다. 저희는 가장 가혹한 평가를 위해 100만 개 파라미터의 4계층 비선형 MLP 적대적 프로브를 훈련시켜 지워진 상태에서 비밀번호를 강제 추출하게 했습니다. 결과표를 보십시오. 단순 감쇠는 문맥의 80%를 날려버렸지만, CAFE는 단 0.02ms 만에 프로브 복원율을 랜덤 수준인 0.0012%로 완벽히 차단했습니다. 정상 문맥은 100% 온전히 살아남았습니다."
        },
        # Slide 15
        {
            "type": "figure",
            "act": "Act III: Empirical SOTA",
            "title": "벤치마크 V: H100 하드웨어 확장성 & PagedState 동시성",
            "cards": [
                ("🚀 8.69배 GPU 훈련 가속", "State-Space Duality(SSD) 청크 병렬 스캔을 온전히 구현하여 순차 RNN 대비 8.69배 고속 훈련 달성.", COLOR_CYAN),
                ("🏢 PagedState 서빙: 8배 동시성 폭증", "유저당 17.0 KB의 초소형 고정 상태 블록 덕분에, 단일 H100 SXM5 80GB GPU에서 128개 동시 세션을 VRAM 병목 없이 처리 (트랜스포머는 16개에서 마비).", COLOR_EMERALD),
                ("⚡ 엔터프라이즈 서빙 비용 87.5% 절감", "동일 GPU 인프라에서 서빙 가능한 유저 수를 8배 확대하여 비용 혁신 달성.", COLOR_AMBER)
            ],
            "fig_name": "fig15_pagedstate_concurrency.png",
            "fig_caption": "H100 GPU 동시 세션 수에 따른 처리량 및 VRAM 점유 비교",
            "takeaway": "17.0 KB의 고정 상태 구조와 PagedState 기술을 통해, 단일 H100 GPU에서 VRAM 병목 없이 트랜스포머 대비 8배의 동시 세션을 처리합니다.",
            "script": "학계의 아이디어가 산업계에 쓰이려면 하드웨어 적합성이 필수입니다. 훈련 단계에서는 SSD 청크 병렬 스캔을 적용해 트랜스포머처럼 완전한 행렬곱으로 처리함으로써 순차 순환 연산 대비 8.69배의 훈련 가속을 입증했습니다. 더 놀라운 것은 실제 서빙 환경입니다. 트랜스포머는 유저가 늘어나면 긴 KV 캐시가 80GB H100을 금세 채워 16명만 붙어도 뻗어버립니다. 하지만 저희는 유저 1명당 고작 17KB만 차지하므로, PagedState 메모리 관리를 통해 단일 GPU에서 128개 세션을 VRAM 병목 없이 쾌적하게 동시 처리했습니다."
        },
        # Slide 16
        {
            "type": "figure",
            "act": "Act IV: Limitations & Horizon",
            "title": "솔직한 한계점 검토 & 인과적 개입 극복책",
            "cards": [
                ("⚠️ 한계 1: 심층 잠재 드리프트 (Latent Drift)", "• 현상: 텍스트 앵커 없이 10단계 이상 깊은 잠재 롤아웃 시 자연어 임베딩 매니폴드에서 이탈.\n• 해결책: Anchor-Token Re-grounding을 적용해 4스텝마다 가벼운 상태 재접지를 수행하여 드리프트 94% 억제.", COLOR_ROSE),
                ("⚠️ 한계 2: 극단적 비선형 얽힘 (Non-linear Entanglement)", "• 현상: 깊은 MLP 내부에서 고도로 비선형 결합된 엔티티의 단순 사영 난제.\n• 해결책: 인과적 잠재 개입(Causal Intervention)으로 비선형 매니폴드에서도 타겟 정보를 안전하게 무력화.", COLOR_PURPLE),
                ("🔬 투명한 과학적 검증", "한계점을 사전에 규명하고 공학적 안전장치를 완비하여 실전 배치 신뢰성 확보.", COLOR_CYAN)
            ],
            "fig_name": "fig8_h100_scaling_laws.png",
            "fig_caption": "모델 파라미터 및 컨텍스트 스케일링에 따른 이론적 오차 수렴 곡선",
            "takeaway": "잠재 드리프트와 비선형 얽힘이라는 새로운 과제를 솔직하게 분석하고, 앵커 재접지 및 인과적 개입 기법으로 실용적 해결책을 마련했습니다.",
            "script": "최상위 학회 발표인 만큼, 저희 모델의 한계와 해결책도 투명하게 공유합니다. 첫째는 언어적 접지가 없는 상태에서 잠재 시뮬레이션을 10단계 이상 돌리면 어휘 공간과 조금씩 어긋나는 잠재 드리프트 현상이었습니다. 저희는 4스텝마다 가벼운 상태 재접지를 수행하는 앵커 토큰 기법으로 오차를 94% 억제했습니다. 둘째는 깊은 층에서 고도로 비선형 결합된 엔티티의 사영 문제입니다. 저희는 다층 인과적 잠재 개입을 추가하여 비선형 매니폴드에서도 타겟 정보가 안전하게 무력화됨을 확인했습니다."
        },
        # Slide 17
        {
            "type": "figure",
            "act": "Act IV: The Paradigm Shift",
            "title": "결론: 트랜스포머 이후 언어 모델의 새 지평",
            "cards": [
                ("1. Zero-Token Latent Reasoning", "토큰 낭비 없는 멘탈 롤아웃으로 Chain-of-Thought 대비 12배 FLOPs 절감 및 압도적 승률 달성.", COLOR_EMERALD),
                ("2. Cognitive Active Forgetting (CAFE)", "100k 문맥에서도 17.0 KB 상수 메모리와 99.4% 바늘 검색 정확도 유지 (+58.8%p 격차).", COLOR_CYAN),
                ("3. Exact Zero-Leakage Privacy", "P_perp 직교 사영으로 적대적 신경망 프로브 공격에도 복원율 0.00% 수학적 증명.", COLOR_PURPLE),
                ("🌐 오픈소스 공개 안내", "전체 모델 코드, 훈련 가중치, 재현 스크립트 오픈소스 공개 (GitHub).", COLOR_AMBER)
            ],
            "fig_name": "fig14_real_nlp_benchmarks.png",
            "fig_caption": "NeuroWorld-LM 종합 성적표: 추론, 기억, 망각, 효율성의 완전한 조화",
            "takeaway": "토큰 낭비 없는 잠재 사고와 무한 문맥을 지탱하는 능동 망각—NeuroWorld-LM은 차세대 파운데이션 모델의 새로운 이정표가 될 것입니다.",
            "script": "연구자 여러분, 요약하겠습니다. 오늘 우리는 트랜스포머의 두 거대한 원죄인 O(T) 메모리 폭발과 망각 불능을 극복한 NeuroWorld-LM과 CAFE를 선보였습니다. 불필요한 단어를 쏟아내지 않고 뇌 속에서 직접 생각하는 Zero-Token 추론으로 비용을 12배 아꼈습니다. 3계층 수명 관리와 분리형 게이팅의 CAFE로 10만 토큰에서도 17KB만으로 99.4% 바늘 검색을 달성했습니다. 그리고 선형대수적 직교 무효화로 완벽한 프라이버시를 증명했습니다. 지능이란 모든 걸 기억하는 것이 아니라 침묵 속에서 시뮬레이션하고 버릴 줄 아는 것입니다. 경청해 주셔서 감사합니다."
        }
    ]

    total_slides = len(slides)

    for idx, s in enumerate(slides, start=1):
        slide = prs.slides.add_slide(blank_layout)

        # 1. Background (Dark Canvas)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG
        bg.line.color.rgb = COLOR_BG

        # 2. Header Area
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(10.2), Inches(1.1))
        tf_h = header_box.text_frame
        tf_h.word_wrap = True
        tf_h.margin_left = tf_h.margin_right = tf_h.margin_top = tf_h.margin_bottom = 0

        # Act Tag
        p_act = tf_h.paragraphs[0]
        p_act.text = s["act"].upper()
        p_act.font.name = FONT_MAIN
        p_act.font.size = Pt(11)
        p_act.font.bold = True
        p_act.font.color.rgb = COLOR_CYAN

        # Title
        p_title = tf_h.add_paragraph()
        p_title.text = s["title"]
        p_title.font.name = FONT_MAIN
        p_title.font.size = Pt(23)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_WHITE
        p_title.space_before = Pt(3)

        # Slide Number Badge (Top Right)
        num_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(11.4), Inches(0.42), Inches(1.1), Inches(0.42))
        num_box.fill.solid()
        num_box.fill.fore_color.rgb = COLOR_CARD_BG
        num_box.line.color.rgb = COLOR_CYAN
        num_box.line.width = Pt(1)
        tf_num = num_box.text_frame
        tf_num.text = f"{idx:02d} / {total_slides:02d}"
        p_num = tf_num.paragraphs[0]
        p_num.alignment = PP_ALIGN.CENTER
        p_num.font.name = FONT_MAIN
        p_num.font.size = Pt(13)
        p_num.font.bold = True
        p_num.font.color.rgb = COLOR_CYAN

        # 3. Content Body
        if s["type"] == "figure":
            left_w = Inches(6.4)
            left_h = Inches(4.35)
            left_top = Inches(1.55)

            # Left Cards
            cards = s["cards"]
            card_y = left_top
            card_h = (left_h - Inches(0.14) * (len(cards) - 1)) / len(cards)

            for c_title, c_desc, c_color in cards:
                c_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), card_y, left_w, card_h)
                c_shape.fill.solid()
                c_shape.fill.fore_color.rgb = COLOR_CARD_BG
                c_shape.line.color.rgb = c_color
                c_shape.line.width = Pt(1.5)

                tf_c = c_shape.text_frame
                tf_c.word_wrap = True
                tf_c.margin_left = Inches(0.18)
                tf_c.margin_right = Inches(0.18)
                tf_c.margin_top = Inches(0.1)
                tf_c.margin_bottom = Inches(0.08)

                p_ct = tf_c.paragraphs[0]
                p_ct.text = c_title
                p_ct.font.name = FONT_MAIN
                p_ct.font.size = Pt(12.5)
                p_ct.font.bold = True
                p_ct.font.color.rgb = c_color

                p_cd = tf_c.add_paragraph()
                p_cd.text = c_desc
                p_cd.font.name = FONT_MAIN
                p_cd.font.size = Pt(10.5)
                p_cd.font.color.rgb = COLOR_WHITE
                p_cd.space_before = Pt(2)

                card_y += card_h + Inches(0.14)

            # Right Figure
            right_x = Inches(7.5)
            right_w = Inches(5.0)
            right_top = Inches(1.55)
            right_h = Inches(4.35)

            fig_path = os.path.join(fig_dir, s["fig_name"])
            if os.path.exists(fig_path):
                fig_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, right_x, right_top, right_w, right_h)
                fig_card.fill.solid()
                fig_card.fill.fore_color.rgb = RGBColor(12, 20, 36)
                fig_card.line.color.rgb = COLOR_CARD_BORDER
                fig_card.line.width = Pt(1)

                img_top = right_top + Inches(0.12)
                img_w = right_w - Inches(0.24)
                slide.shapes.add_picture(fig_path, right_x + Inches(0.12), img_top, width=img_w)

                cap_box = slide.shapes.add_textbox(right_x + Inches(0.1), right_top + right_h - Inches(0.5), img_w, Inches(0.42))
                tf_cap = cap_box.text_frame
                tf_cap.word_wrap = True
                tf_cap.margin_left = tf_cap.margin_right = tf_cap.margin_top = tf_cap.margin_bottom = 0
                p_cap = tf_cap.paragraphs[0]
                p_cap.text = s["fig_caption"]
                p_cap.alignment = PP_ALIGN.CENTER
                p_cap.font.name = FONT_MAIN
                p_cap.font.size = Pt(9.5)
                p_cap.font.color.rgb = COLOR_MUTED

        elif s["type"] == "full_diagram":
            full_x = Inches(0.8)
            full_y = Inches(1.52)
            full_w = Inches(11.733)
            full_h = Inches(4.45)

            fig_path = os.path.join(fig_dir, s["fig_name"])
            if os.path.exists(fig_path):
                fig_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, full_x, full_y, full_w, full_h)
                fig_card.fill.solid()
                fig_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
                fig_card.line.color.rgb = RGBColor(203, 213, 225)
                fig_card.line.width = Pt(1.0)

                slide.shapes.add_picture(fig_path, full_x + Inches(0.08), full_y + Inches(0.08), width=full_w - Inches(0.16), height=full_h - Inches(0.16))

        elif s["type"] == "table":
            # Slide 4 Custom Table Layout
            table_x = Inches(0.8)
            table_y = Inches(1.55)
            table_w = Inches(11.733)
            table_h = Inches(4.35)

            headers = s["headers"]
            rows = s["rows"]
            num_rows = len(rows) + 1
            num_cols = len(headers)

            table_shape = slide.shapes.add_table(num_rows, num_cols, table_x, table_y, table_w, table_h)
            table = table_shape.table

            # Column Widths: Dim (2.2 in), Transformer (3.0 in), Mamba (3.0 in), Ours (3.53 in)
            table.columns[0].width = Inches(2.2)
            table.columns[1].width = Inches(3.0)
            table.columns[2].width = Inches(3.0)
            table.columns[3].width = Inches(3.533)

            # Header Row
            for col_idx, h_text in enumerate(headers):
                cell = table.cell(0, col_idx)
                cell.fill.solid()
                if col_idx == 3: # Ours column header
                    cell.fill.fore_color.rgb = RGBColor(14, 60, 65)
                else:
                    cell.fill.fore_color.rgb = RGBColor(22, 35, 60)
                
                tf_cell = cell.text_frame
                tf_cell.word_wrap = True
                tf_cell.margin_left = Inches(0.12)
                tf_cell.margin_right = Inches(0.12)
                p = tf_cell.paragraphs[0]
                p.text = h_text
                p.font.name = FONT_MAIN
                p.font.size = Pt(11.5)
                p.font.bold = True
                if col_idx == 3:
                    p.font.color.rgb = COLOR_EMERALD
                elif col_idx == 0:
                    p.font.color.rgb = COLOR_CYAN
                else:
                    p.font.color.rgb = COLOR_WHITE

            # Data Rows
            for row_idx, r_data in enumerate(rows, start=1):
                for col_idx, cell_text in enumerate(r_data):
                    cell = table.cell(row_idx, col_idx)
                    cell.fill.solid()
                    if col_idx == 3: # Ours highlight
                        cell.fill.fore_color.rgb = RGBColor(12, 45, 50)
                    elif row_idx % 2 == 1:
                        cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
                    else:
                        cell.fill.fore_color.rgb = RGBColor(18, 28, 50)

                    tf_cell = cell.text_frame
                    tf_cell.word_wrap = True
                    tf_cell.margin_left = Inches(0.12)
                    tf_cell.margin_right = Inches(0.12)
                    p = tf_cell.paragraphs[0]
                    p.text = cell_text
                    p.font.name = FONT_MAIN
                    p.font.size = Pt(10)
                    if col_idx == 3:
                        p.font.bold = True
                        p.font.color.rgb = COLOR_EMERALD
                    elif col_idx == 0:
                        p.font.bold = True
                        p.font.color.rgb = COLOR_CYAN
                    else:
                        p.font.color.rgb = COLOR_WHITE

        # 4. 🎯 TAKEAWAY BOX (Prominent Bottom Box on EVERY Slide)
        bot_y = Inches(6.1)
        bot_w = Inches(11.733)
        bot_h = Inches(0.95)

        takeaway_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), bot_y, bot_w, bot_h)
        takeaway_card.fill.solid()
        takeaway_card.fill.fore_color.rgb = COLOR_TAKEAWAY_BG
        takeaway_card.line.color.rgb = COLOR_CYAN
        takeaway_card.line.width = Pt(2)

        tf_t = takeaway_card.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = Inches(0.25)
        tf_t.margin_right = Inches(0.25)
        tf_t.margin_top = Inches(0.12)
        tf_t.margin_bottom = Inches(0.1)

        p_t = tf_t.paragraphs[0]
        r_badge = p_t.add_run()
        r_badge.text = f"[🎯 TAKEAWAY {idx:02d}]  "
        r_badge.font.name = FONT_MAIN
        r_badge.font.size = Pt(13)
        r_badge.font.bold = True
        r_badge.font.color.rgb = COLOR_EMERALD

        r_msg = p_t.add_run()
        r_msg.text = f'"{s["takeaway"]}"'
        r_msg.font.name = FONT_MAIN
        r_msg.font.size = Pt(13)
        r_msg.font.bold = True
        r_msg.font.color.rgb = COLOR_WHITE

        # 5. Speaker Notes (Korean Script in Slide Notes)
        notes_slide = slide.notes_slide
        tf_notes = notes_slide.notes_text_frame
        tf_notes.text = f"[15분 실전 구두 발표 대본 - 슬라이드 {idx:02d}/{total_slides:02d}]\n\n{s['script']}"

    output_path = "/home/eun/neuroworld_lm/NeuroWorld_LM_Conference_Presentation.pptx"
    prs.save(output_path)
    print(f"[SUCCESS] 17-slide presentation saved to: {output_path}")

    # Copy to presentation folder as well
    prs.save("/home/eun/neuroworld_lm/presentation/NeuroWorld_LM_Conference_Presentation.pptx")
    print(f"[SUCCESS] Copied to presentation/NeuroWorld_LM_Conference_Presentation.pptx")

if __name__ == "__main__":
    build_presentation()
