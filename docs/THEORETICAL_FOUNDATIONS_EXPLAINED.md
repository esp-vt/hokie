# Hokie-LM (NeuroWorld-LM) 이론 및 수학적 정리(Theorems) 완전 정복 가이드

> **대상 독자**: 딥러닝과 선형대수학 기초를 아는 엔지니어부터, "왜 내적(Dot Product)을 쓰는지", "외적(Outer Product)이 무엇인지", "영공간(Null Space)이 왜 프라이버시를 지키는지"에 대해 처음 접하는 비전공자/입문자까지 완전히 이해할 수 있도록 모든 기호와 수식을 밑바닥부터 낱낱이 해설한 문서입니다.

---

# 목차 (Table of Contents)
1. [제1장: 초보자를 위한 선형대수 및 정보이론 핵심 직관 (Math 101)](#제1장-초보자를-위한-선형대수-및-정보이론-핵심-직관-math-101)
   - 1.1 스칼라, 벡터, 행렬, 텐서의 물리적 의미
   - 1.2 내적(Dot Product, $\mathbf{u}^\top \mathbf{v}$)의 본질: "그림자 투영과 유사도"
   - 1.3 외적(Outer Product, $\mathbf{u} \mathbf{v}^\top$)의 본질: "메모리 행렬에 기억 쓰기"
   - 1.4 직교(Orthogonality)와 영공간(Null Space): "완벽한 소거의 비밀"
   - 1.5 행렬 노름(Norm)과 스펙트럼 반지름($\rho(\mathbf{A})$): "시스템이 폭발하지 않는 원리"
   - 1.6 정보이론 3대 기호: 엔트로피 $\mathcal{H}$, KL Divergence $\mathcal{D}_{\mathrm{KL}}$, 상호정보량 $I(X; Y)$
2. [제2장: Hokie-LM 아키텍처 핵심 수식 낱낱이 해부](#제2장-hokie-lm-아키텍처-핵심-수식-낱낱이-해부)
   - 2.1 듀얼 루프 상태 방정식 (Dual-Loop Continuous-Discrete Dynamics)
   - 2.2 서프라이즈 게이트 ($\gamma_t$)와 동적 상태 갱신
   - 2.3 인지 능동 망각 엔진 (CAFE) 3대 수명 채널 ($\boldsymbol{\Omega}$)
   - 2.4 무토큰 잠재 롤아웃 플래너 (Zero-Token Latent Rollout Planner)
3. [제3장: 정리 1 — 서프라이즈 게이팅에 의한 온라인 압축 리그렛 최소화 (Theorem 1)](#제3장-정리-1--서프라이즈-게이팅에-의한-온라인-압축-리그렛-최소화-theorem-1)
   - 3.1 정리의 공식 문장 및 수식
   - 3.2 모든 기호와 변수의 1:1 상세 정의표
   - 3.3 '리그렛(Regret)'이란 무엇이며 왜 sub-linear($\log T$)여야 하는가?
   - 3.4 증명(Proof)의 단계별 쉬운 해설
4. [제4장: 정리 2 — Hurwitz SSM 안정성 하에서의 유계 잠재 드리프트 (Theorem 2)](#제4장-정리-2--hurwitz-ssm-안정성-하에서의-유계-잠재-드리프트-theorem-2)
   - 4.1 정리의 공식 문장 및 수식
   - 4.2 모든 기호와 변수의 1:1 상세 정의표
   - 4.3 '수축 사상(Contraction Mapping)'과 'Hurwitz 안정성'의 직관
   - 4.4 깊은 생각(Rollout $K \to \infty$)을 해도 모델이 왜 미쳐 날뛰지(환각) 않는가?
   - 4.5 증명(Proof)의 단계별 쉬운 해설
5. [제5장: 정리 3 — 트랜스포머의 교란 포화 vs CAFE의 상수 SNR (Theorem 3)](#제5장-정리-3--트랜스포머의-교란-포화-vs-cafe의-상수-snr-theorem-3)
   - 5.1 정리의 공식 문장 및 수식
   - 5.2 모든 기호와 변수의 1:1 상세 정의표
   - 5.3 트랜스포머 Softmax의 치명적 결함: $\text{SNR} \to 0$ (Catastrophic Distraction)
   - 5.4 CAFE는 어떻게 100,000 토큰 후에도 $\text{SNR} \ge C_{\mathrm{min}} > 0$을 유지하는가?
   - 5.5 증명(Proof)의 단계별 쉬운 해설
6. [제6장: 정리 4 (보안 정리 1) — 모델 출력값 역추적 불가능성 (Theorem on Model Inversion Hardness)](#제6장-정리-4-보안-정리-1--모델-출력값-역추적-불가능성-theorem-on-model-inversion-hardness)
   - 6.1 트랜스포머는 왜 출력값으로 프롬프트를 털릴 수 있는가? ($\frac{\partial \hat{\mathbf{y}}_t}{\partial \mathbf{e}_i} \neq 0$)
   - 6.2 Hokie-LM의 다대일(Many-to-One) 압축과 NP-Hard 비선형 게이팅 증명
7. [제7장: 정리 5 (보안 정리 2) — 영공간 직교 투영에 의한 완전 프라이버시 소거 (Theorem on Zero-Leakage)](#제7장-정리-5-보안-정리-2--영공간-직교-투영에-의한-완전-프라이버시-소거-theorem-on-zero-leakage)
   - 7.1 정리의 공식 문장 및 수식
   - 7.2 영공간 사영 행렬 $\mathbf{P}_\perp = \mathbf{I} - \mathbf{V}^\top(\mathbf{V}\mathbf{V}^\top)^{-1}\mathbf{V}$ 유도 원리
   - 7.3 왜 내적 $\langle \mathbf{v}_{\mathrm{secret}}, \mathbf{h}^* \rangle = 0$이 되는가?
   - 7.4 8계층 심층 딥러닝 프로브로 쥐어짜도 상호정보량 $I \equiv 0$인 이유 (Data Processing Inequality)
8. [제8장: 총정리 요약 비교표 (Grand Summary Table)](#제8장-총정리-요약-비교표-grand-summary-table)

---

# 제1장: 초보자를 위한 선형대수 및 정보이론 핵심 직관 (Math 101)

수식을 보기 전에, 인공지능과 본 논문에서 사용하는 수학 기호들의 물리적 실체를 먼저 잡고 들어갑니다.

---

### 1.1 스칼라, 벡터, 행렬, 텐서의 물리적 의미

```
[ 스칼라 (Scalar) ]       [ 벡터 (Vector) ]          [ 행렬 (Matrix) ]              [ 3차원 텐서 (Tensor) ]
     s = 3.14           v = [ 0.2 ]              M = [ 0.2  0.8 ]             T = [ [ 0.2  0.8 ],
                            [ 0.9 ]                  [ 0.5  0.1 ]                   [ 0.5  0.1 ] ],
                            [ 0.1 ]                  [ 0.7  0.4 ]                 [ [ 0.9  0.3 ],
                                                                                    [ 0.4  0.6 ] ]
 (단일 측정값)         (한 단어의 의미 좌표)      (단어들의 관계/지식 맵)        (여러 배치/레이어의 상태)
```

1. **스칼라 (Scalar, 예: $\gamma_t, \beta, \tau$)**: 
   - 그냥 숫자 1개입니다. 예를 들어 "서프라이즈 점수 = 2.4", "온도 = 0.5"처럼 세기나 크기를 나타냅니다.
2. **벡터 (Vector, 보통 굵은 소문자 $\mathbf{x}_t, \mathbf{u}, \mathbf{z}$)**:
   - 숫자들의 1차원 목록입니다. AI에서는 **"한 단어나 개념이 가진 성질들의 좌표"**를 의미합니다.
   - 예: `[사과] = [달콤함: 0.9, 빨간색: 0.8, 컴퓨터: 0.0]`, `[애플회사] = [달콤함: 0.0, 빨간색: 0.1, 컴퓨터: 0.95]`
   - $d_{\mathrm{model}} = 1024$라는 것은 하나의 단어를 1024차원의 공간 좌표로 표현한다는 뜻입니다.
3. **행렬 (Matrix, 보통 굵은 대문자 $\mathbf{A}, \mathbf{B}, \mathbf{W}, \mathbf{P}_\perp$)**:
   - 숫자들이 직사각형 격자(2차원)로 배열된 것입니다. 
   - 벡터를 입력받아 다른 벡터로 변환하는 **"변환 기계(Transform Engine)"** 또는 **"기억 저장소(Memory Table)"** 역할을 합니다.
4. **텐서 (Tensor, 보통 $h_t \in \mathbb{R}^{B \times d_{\mathrm{model}} \times d_{\mathrm{state}}}$)**:
   - 3차원 이상의 다차원 배열입니다. 배치(Batch) $\times$ 특징 차원 $\times$ 상태 차원 등을 한 번에 묶은 것입니다.

---

### 1.2 내적(Dot Product, $\mathbf{u}^\top \mathbf{v}$ 또는 $\langle \mathbf{u}, \mathbf{v} \rangle$)의 본질: "그림자 투영과 유사도"

> **수식 표현**: $\mathbf{u}^\top \mathbf{v} = \langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)$

```
         v
        ^
       /|
      / |
     /  | (수직 수선의 발)
    /   |
   +----+--------> u
     [ u 위에 드리운 v의 그림자 길이 ] = u^T v / ||u||
```

- **직관적 의미**:
  1. 두 벡터 $\mathbf{u}$와 $\mathbf{v}$가 **"얼마나 같은 방향을 바라보고 있는가(유사도, Similarity)"**를 측정합니다.
  2. $\mathbf{u}$와 $\mathbf{v}$의 방향이 완전히 같으면 ($\theta = 0^\circ, \cos(0)=1$) 내적값은 최대가 됩니다.
  3. $\mathbf{u}$와 $\mathbf{v}$가 서로 수직(직교)이면 ($\theta = 90^\circ, \cos(90)=0$) **내적값은 정확히 $0$**이 됩니다.
  4. 본 논문에서 프라이버시 삭제를 증명할 때 $\langle \mathbf{v}_{\mathrm{secret}}, \mathbf{h}^* \rangle = 0$이 나오는 이유는, 지워야 할 비밀 방향($\mathbf{v}$)과 모델의 남은 기억($\mathbf{h}^*$)이 **서로 완벽히 수직(직교)이라 어떤 그림자도 남지 않기 때문**입니다.

---

### 1.3 외적(Outer Product, $\mathbf{u} \mathbf{v}^\top$)의 본질: "메모리 행렬에 기억 쓰기"

> **수식 표현**: 
> $\mathbf{u} \in \mathbb{R}^N, \mathbf{v} \in \mathbb{R}^M$ 일 때, $\mathbf{u} \mathbf{v}^\top \in \mathbb{R}^{N \times M}$ (행렬이 됨)

```
       v^T = [ v1    v2    v3 ]
 u = [ u1 ]  [ u1*v1 u1*v2 u1*v3 ]
     [ u2 ]  [ u2*v1 u2*v2 u2*v3 ]  <--- N x M 크기의 새로운 지식 행렬 생성!
     [ u3 ]  [ u3*v1 u3*v2 u3*v3 ]
```

- **직관적 의미**:
  - 내적($\mathbf{u}^\top \mathbf{v}$)은 두 벡터를 곱해 **숫자 1개(스칼라)**를 만들지만,
  - 외적($\mathbf{u} \mathbf{v}^\top$)은 두 벡터를 곱해 **거대한 행렬(Matrix)**을 만듭니다.
  - 연상 기억(Associative Memory)에서 Key 벡터 $\mathbf{k}$와 Value 벡터 $\mathbf{v}$를 연결하여 메모리 행렬 $\mathbf{M}$에 저장할 때 $\mathbf{M} \leftarrow \mathbf{M} + \mathbf{v} \mathbf{k}^\top$ 형태로 씁니다.

---

### 1.4 직교(Orthogonality)와 영공간(Null Space): "완벽한 소거의 비밀"

```
                    y축 (Secret Subspace: 비밀 정보 축)
                     ^
                     |   * [비밀 벡터 v_secret]
                     |   
                     |   
  -------------------+----------------------------> x축 (Orthogonal Complement: 일반 지식 축)
                     |
                     |   * [소거 후 남은 상태 h* = P_perp * h]
                     |
                     (y축 성분이 정확히 0이 됨!)
```

1. **직교(Orthogonal, $\perp$)**: 두 벡터가 $90^\circ$를 이루어 서로 전혀 영향을 주지 않는 상태입니다.
2. **사영(Projection, $\mathbf{P}$)**: 어떤 벡터를 특정 평면 위에 빛을 비춰 그림자를 떨어뜨리는 연산입니다.
3. **직교 여공간 사영(Orthogonal Complement Projection, $\mathbf{P}_\perp = \mathbf{I} - \mathbf{P}$)**:
   - 전체 공간($\mathbf{I}$)에서 "비밀 정보가 사는 공간($\mathbf{P}$)"을 **통째로 빼버리는 연산자**입니다.
   - 어떤 기억 $\mathbf{h}$에 $\mathbf{P}_\perp$를 곱하면, 비밀 정보 방향 성분은 **마치 가위로 오려낸 것처럼 정확히 $0$**이 되고, 비밀과 무관한 일반 지식은 $100\%$ 그대로 남습니다.

---

### 1.5 행렬 노름(Norm)과 스펙트럼 반지름($\rho(\mathbf{A})$): "시스템이 폭발하지 않는 원리"

1. **벡터 노름 ($\|\mathbf{x}\|$)**: 벡터의 물리적 길이입니다. 예: 2차원에서 $\sqrt{x_1^2 + x_2^2}$.
2. **행렬 노름 ($\|\mathbf{A}\|$)**: 행렬 $\mathbf{A}$가 벡터를 최대 몇 배까지 잡아 늘릴 수 있는가의 배율입니다.
3. **스펙트럼 반지름 ($\rho(\mathbf{A})$)**: 행렬 $\mathbf{A}$의 고윳값(Eigenvalue) 중 가장 큰 절댓값입니다.
   - 만약 $\rho(\mathbf{A}) \ge 1$ 이면: 루프를 돌 때마다 상태가 눈덩이처럼 불어나 **무한대로 폭발(Explosion)**합니다.
   - 만약 $\rho(\mathbf{A}) \le 1 - \epsilon < 1$ 이면: 루프를 $K$번 돌릴 때마다 $(1-\epsilon)^K \to 0$으로 줄어들어 **시스템이 스스로 진정(Contraction, 수축)**됩니다.

---

### 1.6 정보이론 3대 기호: 엔트로피, KL Divergence, 상호정보량

1. **엔트로피 ($\mathcal{H}(X)$)**:
   - 확률 변수 $X$가 가진 **"불확실성(깜짝 놀람의 잠재량)"**입니다. 동전 던지기는 불확실성이 높고, 항상 1만 나오는 주사위는 엔트로피가 0입니다.
2. **KL Divergence ($\mathcal{D}_{\mathrm{KL}}(q \parallel p)$)**:
   - 두 확률분포 $q$와 $p$ 사이의 **"정보적 괴리(놀라움의 크기, Surprise)"**입니다.
   - 내 뇌의 사전 예측($p$)과 눈앞에 나타난 실제 단어($q$)가 완벽히 일치하면 $\mathcal{D}_{\mathrm{KL}} = 0$ (하나도 안 놀람).
   - 갑자기 엉뚱한 외계어 단어가 튀어나오면 $\mathcal{D}_{\mathrm{KL}} \gg 0$ (서프라이즈 폭발 $\rightarrow$ 메모리에 강하게 기록!).
3. **상호정보량 ($I(X; Y)$)**:
   - $Y$라는 변수를 관측했을 때, $X$에 대해 알게 되는 **"공유 정보의 양(비트 수)"**입니다.
   - $I(X; Y) = 0$ 이라는 것은: $Y$를 아무리 현미경으로 뜯어보고 100층짜리 딥러닝으로 분석해도 **$X$에 대한 단 $1\text{ bit}$의 단서도 얻을 수 없음(완벽한 독립, Absolute Zero-Leakage)**을 의미합니다.

---

# 제2장: Hokie-LM 아키텍처 핵심 수식 낱낱이 해부

논문의 Section 3(Methodology)에 등장하는 핵심 수식 체계를 변수 단위로 해부합니다.

```
                   [ 입력 토큰 x_t ]
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[ 사후 네트워크 q_phi ]           [ 사전 네트워크 p_theta ]
q(z_t | h_{t-1}, x_t)            p(z_t | h_{t-1})
         │                                 │
         └────────────────┬────────────────┘
                          ▼
            [ KL 서프라이즈 계산 ]
            γ_t = D_KL( q || p )
                          │
          ┌───────────────┴───────────────┐
          │ (Salience Boost)              │ (Eviction Inhibit)
          ▼                               ▼
 [ 입력 증폭: (1 + σ(W_s γ_t)) ]     [ 망각 억제: E_t → 0 ]
          │                               │
          └───────────────┬───────────────┘
                          ▼
             [ 연속-이산 SSM 듀얼 루프 ]
     h_t = A_t h_{t-1} + B_t [e(x_t); z_t]
                          │
                          ▼
             [ CAFE 영공간 소거 (P_⊥) ]
                 h*_t = P_⊥ h_t
```

---

### 2.1 듀얼 루프 상태 방정식 (식 1 ~ 4)

$$\begin{aligned}
p_\theta(z_t \mid h_{t-1}) &= \mathrm{Cat}\big(\mathrm{Softmax}(W_{\mathrm{prior}} h_{t-1})\big) \quad &\text{--- [식 1: 사전 예측]} \\
q_\phi(z_t \mid h_{t-1}, x_t) &= \mathrm{Cat}\big(\mathrm{Softmax}(W_{\mathrm{post}} [h_{t-1}; e(x_t)])\big) \quad &\text{--- [식 2: 사후 인식]} \\
\gamma_t &\triangleq \mathcal{D}_{\mathrm{KL}}\big( q_\phi \,\parallel\, p_\theta \big) \quad &\text{--- [식 3: 정보적 서프라이즈]} \\
h_t^{\mathrm{raw}} &= \mathbf{A}_t^{\mathrm{active}} h_{t-1} + \mathbf{B}_t \Big( \big(1 + \sigma(W_s \gamma_t)\big) \cdot [e(x_t); z_t] \Big) \quad &\text{--- [식 4: 상태 갱신]}
\end{aligned}$$

#### 🔍 각 변수의 의미:
- $h_{t-1} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{state}}}$: 과거 $t-1$ 시점까지의 모든 연속적인 문맥이 압축된 **연속형 SSM 상태** (크기: $17.0\text{ KB}$).
- $x_t \in \{1, \dots, V\}$: 현재 들어온 입력 단어(토큰) 번호.
- $e(x_t) \in \mathbb{R}^{d_{\mathrm{model}}}$: 입력 단어 $x_t$를 다차원 벡터 공간에 매핑한 **임베딩 벡터**.
- $p_\theta(z_t \mid h_{t-1})$: 과거 문맥($h_{t-1}$)만 보고 "다음에 어떤 개념($z_t$)이 올까?" 하고 뇌가 **스스로 상상(예측)한 사전 확률분포**.
- $q_\phi(z_t \mid h_{t-1}, x_t)$: 과거 문맥($h_{t-1}$)에 더해 실제 눈앞의 단어($e(x_t)$)까지 보고 "아, 이 단어는 이런 뜻이구나!" 하고 **정확히 인식한 사후 확률분포**.
- $z_t \in \Delta^{32 \times 32}$: $32$개 범주 각각에서 $32$개 클래스 중 하나를 선택하는 **이산 범주형(Categorical) 잠재 변수** (뇌의 고수준 개념 심볼).
- $\gamma_t \ge 0$: 사후 분포($q_\phi$)와 사전 분포($p_\theta$) 간의 KL 발산으로 측정한 **"정보적 새로움/놀라움(Surprise)"**.
- $\sigma(W_s \gamma_t) \in (0, 1)$: 서프라이즈 점수 $\gamma_t$를 Sigmoid 함수를 거쳐 $0 \sim 1$ 사이의 게이트 가중치로 변환한 것.
- $[e(x_t); z_t]$: 표면 단어 벡터 $e(x_t)$와 고수준 개념 벡터 $z_t$를 세로로 이어 붙인(Concatenate) 통합 표현.

---

### 2.2 인지 능동 망각 엔진 (CAFE) 수식 체계

$$\boldsymbol{\Omega} = \big[ \underbrace{0.05 \cdot \mathbf{1}^{d_{\mathrm{pers}}}}_{\text{영구 채널 (60\%)}}, \; \underbrace{1.0 \cdot \mathbf{1}^{d_{\mathrm{work}}}}_{\text{작업 채널 (30\%)}}, \; \underbrace{30.0 \cdot \mathbf{1}^{d_{\mathrm{scratch}}}}_{\text{스크래치패드 (10\%)}} \big]^\top$$

$$\mathbf{A}_t^{\mathrm{active}} = \exp\left( -\Delta_t \mathbf{A} \odot (1 + \boldsymbol{\Omega} \odot E_t) \right)$$

$$\mathbf{P}_{\perp} = \mathbf{I} - \frac{\tilde{v}_{\mathrm{obs}} \tilde{v}_{\mathrm{obs}}^\top}{\|\tilde{v}_{\mathrm{obs}}\|^2}, \quad h_t = \mathbf{P}_{\perp} h_t^{\mathrm{raw}}$$

#### 🔍 각 변수의 의미:
- $\boldsymbol{\Omega}$ (오메가): 상태 메모리 채널들의 **수명 감쇄 계수 벡터**.
  - 영구 채널 ($60\%$): $\omega = 0.05 \to$ 거의 영원히 잊지 않고 보존 (시스템 프롬프트, 인물 페르소나).
  - 작업 채널 ($30\%$): $\omega = 1.0 \to$ 현재 대화 문맥 유지.
  - 스크래치패드 ($10\%$): $\omega = 30.0 \to$ 계산이 끝나면 즉시 휘발 (단기 암산, 일회용 툴 로그).
- $E_t \in [0, 1]$: **능동 방출(Eviction) 게이트**. 새로운 중요한 정보가 들어오면($\gamma_t \uparrow$) $E_t \to 0$이 되어 망각을 차단하고, 토픽이 바뀌면 $E_t \to 1$이 되어 잡음을 방출.
- $\mathbf{P}_\perp$: 특정 변수($v_{\mathrm{obs}}$)를 모델의 모든 기억에서 흔적도 없이 도려내는 **영공간 직교 투영 행렬**.

---

# 제3장: 정리 1 — 서프라이즈 게이팅에 의한 온라인 압축 리그렛 최소화 (Theorem 1)

---

### 3.1 정리의 공식 문장 및 수식

> **Theorem 1 (Variational Regret Bound with Surprise Gating)**  
> 시퀀스 데이터 분포 $p_{\mathrm{data}}(x_{1:T})$가 $M = |\mathcal{T}^*|$개의 의미적 변곡점(Change-points) $\mathcal{T}^* = \{t_1^*, \dots, t_M^*\}$을 가진다고 하자.  
> 서프라이즈 변조 갱신 $\mathbf{u}_{\mathrm{eff}} = (1 + \sigma(W_s \gamma_t)) \mathbf{u}_t$ (단, $\gamma_t = \mathcal{D}_{\mathrm{KL}}(q_\phi \parallel p_\theta)$) 하에서, 모델의 누적 시퀀스 예측 리그렛(Cumulative Regret) $\mathcal{R}_T$는 다음 상계를 만족한다:
>
> $$\mathcal{R}_T \triangleq \sum_{t=1}^T \left[ -\log P(x_t \mid h_{t-1}) - \min_{\theta^*} \mathbb{E}[-\log P_{\theta^*}(x_t \mid x_{<t})] \right] \le \mathcal{O}\left( |\mathcal{T}^*| \cdot d_{\mathrm{model}} \log T + \sum_{t \notin \mathcal{T}^*} \gamma_t \right)$$

---

### 3.2 모든 기호와 변수의 1:1 상세 정의표

| 기호 / 수식 | 수학적 정의 | 단위 / 형태 | 물리적·직관적 의미 |
| :--- | :--- | :--- | :--- |
| $T$ | 시퀀스 전체 길이 | 자연수 ($T \in \mathbb{N}$) | 전체 토큰 수 (예: 16,384 토큰, 100,000 토큰) |
| $x_{1:T} = [x_1, \dots, x_T]$ | 입력 토큰 시퀀스 | 단어 ID 배열 | 문장 또는 문서 전체의 단어 나열 |
| $\mathcal{T}^* = \{t_1^*, \dots, t_M^*\}$ | 변곡점 집합 | 시점 인덱스 집합 | 새로운 주제가 시작되거나 중요한 사건이 터진 시점들 |
| $M = |\mathcal{T}^*|$ | 변곡점의 총 개수 | 자연수 ($M \ll T$) | 10만 단어 중 진짜 중요한 사건이 일어난 횟수 (예: 100번) |
| $-\log P(x_t \mid h_{t-1})$ | 온라인 음의 로그 우도 | 실수 ($\ge 0$, nats/bits) | $17\text{ KB}$ 상수 상태 $h_{t-1}$만 보고 다음 단어 $x_t$를 예측할 때 느끼는 **당혹감(Loss)** |
| $\min_{\theta^*} \mathbb{E}[-\log P_{\theta^*}(x_t \mid x_{<t})]$ | 전지전능한 이상적 오라클의 Loss | 실수 ($\ge 0$) | 무한한 KV 캐시를 다 보고 사후에 구한 **이론상 완벽한 최적의 Loss** |
| $\mathcal{R}_T$ (리그렛, Regret) | 두 손실(Loss)의 누적 차이 | 실수 ($\ge 0$) | **"상수 메모리로 압축해서 예측하느라 완벽한 오라클에 비해 누적해서 손해 본 총량"** |
| $d_{\mathrm{model}}$ | 은닉층 차원 수 | 자연수 (512 또는 1024) | 단어 표현의 벡터 차원 수 |
| $\sum_{t \notin \mathcal{T}^*} \gamma_t$ | 평온한 구간에서의 서프라이즈 합 | 실수 ($\ge 0$) | 주제가 안 바뀌는 평온한 일상 구간에서의 미세 잔여 오차 |

---

### 3.3 '리그렛(Regret)'이란 무엇이며 왜 sub-linear($\log T$)여야 하는가?

- **비유**: 여러분이 100권짜리 책을 읽으면서, 책 전체를 통째로 외우는 천재(오라클, Transformer)와 손바닥만 한 메모지 1장($17\text{ KB}$)에 요약하며 읽는 여러분(Hokie-LM)의 시험 점수 차이를 비교합니다.
- 만약 여러분의 요약 방식이 엉망이면, 책이 100권으로 늘어날 때 손해 보는 점수 차이($\mathcal{R}_T$)는 책 권수 $T$에 비례하여 $\mathcal{O}(T)$로 선형 폭발(Linear Explosion)합니다. $\to$ **기억 상실 발생!**
- 하지만 Theorem 1은 우리의 손해 총량이 오직 **"주제가 바뀐 횟수 $M$"**에만 비례하고, 시간 $T$에 대해서는 매우 완만한 $\log T$로만 증가함을 증명합니다 ($\mathcal{O}(M \log T)$).
- 따라서 시간당 평균 리그렛은 $\lim_{T \to \infty} \frac{\mathcal{R}_T}{T} = 0$이 되어, **아무리 긴 글을 읽어도 상수 메모리 $17\text{ KB}$로 인한 정보 손실이 영(Zero)으로 수렴함**을 수학적으로 보장합니다.

---

### 3.4 증명(Proof)의 단계별 쉬운 해설

1. **구간 분할 (Regime Partitioning)**:
   전체 $1 \sim T$ 시간을 변곡점 $t_k^*$를 기준으로 $M+1$개의 평온한 구간(Stationary Epochs)으로 나눕니다.
2. **평온한 구간 ($t \notin \mathcal{T}^*$)에서의 오차 유계**:
   주제가 안 바뀌는 동안에는 다음 단어가 예측 가능하므로 서프라이즈 $\gamma_t = \mathcal{D}_{\mathrm{KL}} \le \epsilon$입니다. 상태 갱신이 안정적이므로 이 구간의 누적 오차는 $\sum \gamma_t$로 제한됩니다.
3. **변곡점 ($t \in \mathcal{T}^*$)에서의 정보 흡수**:
   새로운 중요한 정보가 등장하면 서프라이즈 $\gamma_t \gg 0$가 스파이크를 칩니다. 이때 식 (4)의 증폭 계수 $(1 + \sigma(W_s \gamma_t))$가 즉각 작동하여 새로 들어온 핵심 정보를 $h_t$에 강하게 새깁니다. 이 전환 비용은 정보이론적 MDL(Minimum Description Length) 원리에 의해 $\mathcal{O}(d_{\mathrm{model}} \log T)$로 유계됩니다.
4. **합산**: $M$개의 변곡점 비용과 평온 구간 비용을 합치면 정리 1의 최종 상계 $\mathcal{O}(M \cdot d_{\mathrm{model}} \log T + \sum \gamma_t)$가 완성됩니다. $\blacksquare$

---

# 제4장: 정리 2 — Hurwitz SSM 안정성 하에서의 유계 잠재 드리프트 (Theorem 2)

---

### 4.1 정리의 공식 문장 및 수식

> **Theorem 2 (Bounded Latent Drift under Hurwitz SSM Stability)**  
> 이산화된 시스템 행렬의 스펙트럼 반지름이 $\rho(\mathbf{A}_t) \le 1 - \epsilon$ (단, $\epsilon \in (0, 1)$)을 만족하는 대각 Hurwitz 안정성을 가진다고 가정하자.  
> $L_z = \|\mathbf{B}_z\|$를 잠재 변환의 립시츠 연속 상수(Lipschitz constant)라 하자.  
> 임의의 초기 상태 $h_t$와 임의의 무제한 롤아웃 깊이 $K \ge 1$에 대하여 다음 부등식이 성립한다:
>
> $$\|h_{t+K}\| \le (1 - \epsilon)^K \|h_t\| + \frac{1 - (1 - \epsilon)^K}{\epsilon} L_z \|z_{\mathrm{max}}\|$$
>
> 나아가, 롤아웃 깊이가 무한대로 갈 때의 점근적 상태 드리프트(Asymptotic Drift)는 엄격히 유계된다:
>
> $$\lim_{K \rightarrow \infty} \|h_{t+K} - h_t\| \le \frac{1}{\epsilon} L_z \|z_{\mathrm{max}}\| < \infty$$

---

### 4.2 모든 기호와 변수의 1:1 상세 정의표

| 기호 / 수식 | 수학적 정의 | 단위 / 형태 | 물리적·직관적 의미 |
| :--- | :--- | :--- | :--- |
| $K$ | 잠재 생각 롤아웃 깊이 | 자연수 ($K = 1, 2, \dots, \infty$) | 텍스트 출력 없이 머릿속으로 시뮬레이션한 생각의 단계 수 |
| $h_t$ | $t$ 시점의 초기 상태 벡터 | 벡터 ($\mathbb{R}^{d_{\mathrm{model}}}$) | 생각을 시작하기 직전의 현재 기억 상태 |
| $h_{t+K}$ | $K$단계 상상 후의 상태 벡터 | 벡터 ($\mathbb{R}^{d_{\mathrm{model}}}$) | 머릿속으로 $K$번 시뮬레이션한 미래의 생각 상태 |
| $\mathbf{A}_t$ | SSM 전이 행렬 (Transition Matrix) | 대각 행렬 ($\mathbb{R}^{d \times d}$) | 과거 상태를 다음 상태로 전달할 때의 감쇄/전이 연산자 |
| $\rho(\mathbf{A}_t)$ | 행렬 $\mathbf{A}_t$의 스펙트럼 반지름 | 실수 ($0 < \rho < 1$) | 한 번 생각할 때 과거 기억이 줄어드는 감쇄율 (예: 0.95) |
| $\epsilon$ | 안정성 마진 (Stability Margin) | 실수 ($\epsilon \in (0, 1)$) | 감쇄율이 1보다 얼마나 작은지를 나타내는 안전 마진 ($1 - \rho$) |
| $\mathbf{B}_z$ | 잠재 입력 투영 행렬 | 행렬 ($\mathbb{R}^{d \times d_z}$) | 개념 코드 $z$를 연속 상태 $h$로 주입하는 연결 행렬 |
| $L_z = \|\mathbf{B}_z\|$ | 립시츠 상수 (Lipschitz Constant) | 실수 ($L_z > 0$) | 개념 $z$가 변할 때 상태 $h$가 최대 몇 배 변할 수 있는가의 민감도 |
| $z_{\mathrm{max}}$ | 잠재 심볼 벡터의 최댓값 | 벡터 ($\|z\| \le 1$) | Gumbel-Softmax에 의해 확률 심플렉스($\sum z_i = 1$) 내로 제한된 개념 크기 |
| $\|h_{t+K} - h_t\|$ | 생각의 총 이동 거리 (Drift) | 실수 ($\ge 0$) | 처음 생각에서 얼마나 멀리 떨어져 나갔는가의 거리 |

---

### 4.3 '수축 사상(Contraction Mapping)'과 'Hurwitz 안정성'의 직관

```
   [ 초기 상태 h_t ]
         │ (x 0.9)
         ▼
   [ 1단계 h_{t+1} ] ───> + 새 생각 주입 (크기 1.0)
         │ (x 0.9)
         ▼
   [ 2단계 h_{t+2} ] ───> + 새 생각 주입 (크기 1.0)
         │
         ▼
       ..... (K번 반복)
         │
         ▼
   [ 수렴 한계선: 1.0 / (1 - 0.9) = 10.0 이하로 절대 못 벗어남! ]
```

- **직관적 비유 (용수철과 감쇄기)**:
  - Hurwitz 안정성이란 진동하는 용수철에 끈적한 오일 댐퍼(감쇄기)를 달아놓은 것과 같습니다.
  - 아무리 밖에서 발로 차고 흔들어도(새로운 생각 $z$를 계속 집어넣어도), 오일 저항($(1-\epsilon)$) 때문에 시스템은 특정 반경 밖으로 튕겨 나가지 못합니다.
  - 이것이 수학에서 말하는 **바나흐 고정점 정리(Banach Fixed Point Theorem / Contraction Mapping)**의 원리입니다.

---

### 4.4 깊은 생각(Rollout $K \to \infty$)을 해도 모델이 왜 미쳐 날뛰지 않는가?

- 기존 생성 모델의 최대 공포: "머릿속으로 계속 상상만 돌리면(Ungrounded Rollout), 오차가 누적되어 결국 외계어를 뱉거나 헛소리(Hallucination Explosion)를 하지 않을까?"
- **Theorem 2의 철퇴**:
  - $K \to \infty$로 갈 때, 초기 상태 $h_t$의 영향력은 $(1-\epsilon)^K \to 0$으로 완전히 소멸합니다.
  - 새로 들어오는 생각들의 누적 합은 등비급수 $\sum_{k=0}^\infty (1-\epsilon)^k = \frac{1}{\epsilon}$으로 **정확히 상한선에 수렴**합니다.
  - 따라서 100번, 10,000번 깊은 생각을 돌려도 모델의 잠재 상태는 안전한 구체(Attractor Basin) 안에 갇혀 있어 환각 폭발이 원천적으로 불가능합니다.

---

### 4.5 증명(Proof)의 단계별 쉬운 해설

1. **재귀 전개 (Recurrence Unrolling)**:
   $h_{t+K} = \mathbf{A}^K h_t + \sum_{k=0}^{K-1} \mathbf{A}^k \mathbf{B}_z z_{t+K-k}$
2. **삼각 부등식 및 놈 성질 적용**:
   $\|h_{t+K}\| \le \|\mathbf{A}\|^K \|h_t\| + \|\mathbf{B}_z\| \|z_{\mathrm{max}}\| \sum_{k=0}^{K-1} \|\mathbf{A}\|^k$
3. **스펙트럼 상계 $\rho(\mathbf{A}) \le 1-\epsilon$ 대입**:
   등비수열의 합 공식 $\sum_{k=0}^{K-1} (1-\epsilon)^k = \frac{1 - (1-\epsilon)^K}{\epsilon}$을 적용합니다.
4. **극한 취하기 ($K \to \infty$)**:
   $(1-\epsilon)^K \to 0$ 이므로, $\|h_{t+K}\| \le \frac{L_z \|z_{\mathrm{max}}\|}{\epsilon} < \infty$ 로 엄격히 유계됩니다. $\blacksquare$

---

# 제5장: 정리 3 — 트랜스포머의 교란 포화 vs CAFE의 상수 SNR (Theorem 3)

---

### 5.1 정리의 공식 문장 및 수식

> **Theorem 3 (Distractor Saturation in Transformers vs. Constant SNR in CAFE)**  
> 시퀀스 길이 $T$ 내에 유의미한 정보 토큰 $N_{\mathrm{sal}}$개와 무의미한 교란 토큰(Distractors) $T - N_{\mathrm{sal}}$개가 섞여 있다고 하자.  
> Softmax Attention 하에서 신호 대 잡음비(Signal-to-Noise Ratio)는 시퀀스 길이가 증가함에 따라 $0$으로 붕괴한다:
>
> $$\lim_{T \rightarrow \infty} \mathrm{SNR}_{\mathrm{Transformer}}(T) \triangleq \lim_{T \rightarrow \infty} \frac{\mathbb{E}[Z_{\mathrm{salient}}]}{\mathbb{E}[Z_{\mathrm{distractor}}]} = 0 \quad \text{(Catastrophic Distraction)}$$
>
> 반면, 다중 스케일 채널 분할 $\boldsymbol{\Omega}$ 및 능동 방출 $E_t$를 적용한 CAFE 하에서는 교란 상태 에너지가 지수적으로 억제되고 유의미한 채널의 전이 계수는 $\mathbf{A}_{\mathrm{sal}}^{\mathrm{active}} \approx 1$을 유지하여, SNR이 엄격한 양수 상수로 하한 유계된다:
>
> $$\lim_{T \rightarrow \infty} \mathrm{SNR}_{\mathrm{CAFE}}(T) \ge \frac{N_{\mathrm{sal}} \cdot \|\mathbf{B}_{\mathrm{sal}}\|}{\frac{1}{1 - \rho(\mathbf{A}_{\mathrm{dist}})} \|\mathbf{B}_{\mathrm{dist}}\|} \ge C_{\mathrm{min}} > 0$$

---

### 5.2 모든 기호와 변수의 1:1 상세 정의표

| 기호 / 수식 | 수학적 정의 | 단위 / 형태 | 물리적·직관적 의미 |
| :--- | :--- | :--- | :--- |
| $N_{\mathrm{sal}}$ | 핵심 정보 토큰의 수 (Salient) | 자연수 (상수) | 기억해야 할 진짜 중요한 단어 수 (예: 비밀번호, 핵심 룰 10개) |
| $T - N_{\mathrm{sal}}$ | 잡음 교란 토큰의 수 (Distractor) | 자연수 ($T \to \infty$) | 중간에 지나가는 수만 개의 잡담, 일회용 로그, 무의미한 단어들 |
| $Z_{\mathrm{salient}}$ | 핵심 정보에 배정된 어텐션 점수 | 실수 ($\ge 0$) | 모델이 중요한 정보에 쏟는 관심의 총합 |
| $Z_{\mathrm{distractor}}$ | 교란 잡음에 분산된 어텐션 점수 | 실수 ($\ge 0$) | 모델이 엉뚱한 잡음에 낭비하는 관심의 총합 |
| $\mathrm{SNR}(T)$ | 신호 대 잡음비 (Signal-to-Noise Ratio) | 무차원 비율 (또는 $\text{dB}$) | **"잡음 대비 진짜 핵심 기억이 얼마나 또렷하게 살아있는가?"** |
| $\mathbf{A}_{\mathrm{sal}}^{\mathrm{active}}$ | 핵심 채널의 보존 계수 | 실수 ($\approx 1.0$) | CAFE 영구 채널($\omega=0.05$)에서 핵심 정보가 안 지워지고 유지되는 비율 |
| $\rho(\mathbf{A}_{\mathrm{dist}})$ | 교란 채널의 감쇄율 | 실수 ($< 1.0$) | CAFE 스크래치패드 채널($\omega=30.0$)에서 잡음이 빠르게 날아가는 비율 |
| $C_{\mathrm{min}}$ | CAFE가 보장하는 최소 SNR 바운드 | 양의 실수 ($C_{\mathrm{min}} > 0$) | 100만 단어가 지나도 절대 깨지지 않는 최소 기억 선명도 (실측 $24.7\text{ dB}$) |

---

### 5.3 트랜스포머 Softmax의 치명적 결함: 왜 $\text{SNR} \to 0$으로 망가지는가?

```
[ 트랜스포머의 어텐션 소프트맥스 분모 ]
 분모 = exp(중요정보 1) + exp(중요정보 2) + exp(잡음 1) + exp(잡음 2) + ... + exp(잡음 10,000)
                                      └──────────────────────────────────────────────┘
                                          잡음 1만 개가 쌓이면 분모가 무한대로 폭발!
                                          결과: 중요정보 1개의 비중 = 1 / 10,000 → 0%로 희석!
```

- Softmax 함수는 모든 입력에 대해 $\exp(x) > 0$ (엄격한 양수)를 출력합니다.
- 즉, 아무리 쓸모없는 쓰레기 토큰이라도 어텐션 가중치를 최소 $0.0001$이라도 가져갑니다.
- 시퀀스 길이 $T$가 1만, 10만으로 길어지면, 잡음 토큰 수천 개가 가져가는 가중치의 합($\sum \exp$)이 분모를 장악하여, **정작 10만 단어 전에 정의한 시스템 프롬프트의 어텐션 가중치가 $0$으로 희석**되어 버립니다. (Context Rot / Catastrophic Distraction 발생!)

---

### 5.4 CAFE는 어떻게 100,000 토큰 후에도 $\text{SNR} \ge C_{\mathrm{min}} > 0$을 유지하는가?

- **CAFE의 비결: 채널 격리 + 능동 망각**:
  1. 핵심 룰은 영구 채널($\boldsymbol{\Omega}_{\mathrm{pers}}$)에 넣어 감쇄율 $\mathbf{A}_{\mathrm{sal}} \approx 1.0$으로 영구 락을 겁니다.
  2. 일회용 잡음은 스크래치 채널($\boldsymbol{\Omega}_{\mathrm{scratch}}$)에 넣어 들어오자마자 $\mathbf{A}_{\mathrm{dist}} \ll 1$로 광속 감쇄시켜 버립니다.
  3. 잡음이 아무리 $T \to \infty$로 쏟아져도, 감쇄 등비급수 공식에 의해 잡음의 총에너지는 $\frac{1}{1-\rho(\mathbf{A}_{\mathrm{dist}})} \|\mathbf{B}_{\mathrm{dist}}\|$라는 **작은 유한한 값으로 상한**이 걸립니다.
  4. 따라서 분자(핵심 기억)는 보존되고 분모(잡음)는 유계되므로, $\mathrm{SNR} \ge C_{\mathrm{min}} = 24.7\text{ dB}$가 영원히 평평하게 유지됩니다.

---

# 제6장: 정리 4 (보안 정리 1) — 모델 출력값 역추적 불가능성 (Theorem on Model Inversion Hardness)

---

### 6.1 트랜스포머는 왜 출력값으로 프롬프트를 털릴 수 있는가?

$$\frac{\partial \hat{\mathbf{y}}_t}{\partial \mathbf{e}_i} = \alpha_{t,i} \mathbf{W}_{\mathrm{head}} \frac{\partial \mathrm{LN}}{\partial \mathbf{h}_t} \mathbf{W}_O^\top \mathbf{W}_V^\top + \mathcal{O}\left( \frac{\partial \alpha_{t,i}}{\partial \mathbf{e}_i} \right) \neq \mathbf{0}$$

- 트랜스포머는 KV 캐시에 과거 모든 단어 $x_i$의 임베딩 $\mathbf{e}_i$를 압축 없이 날것 그대로 들고 있습니다.
- 따라서 출력 단어 $\hat{\mathbf{y}}_t$에서 입력 단어 $\mathbf{e}_i$로의 편미분 그래디언트($\frac{\partial \hat{\mathbf{y}}_t}{\partial \mathbf{e}_i}$)가 **항상 $0$이 아닌 유의미한 값**을 갖습니다.
- 공격자는 출력 확률값만 보고 역전파 경사하강법($\arg\min \|\mathcal{M}(\tilde{\mathbf{e}}) - \hat{\mathbf{y}}\|^2$)을 돌리면, **사용자가 처음에 입력한 비밀 프롬프트를 고스란히 복원**해 낼 수 있습니다.

---

### 6.2 Hokie-LM의 다대일(Many-to-One) 압축과 NP-Hard 비선형 게이팅 증명

```
[ 입력 공간: V^L (경우의 수 50,000^16,384 = 무한대) ]
                   │
                   ▼  (손실 압축 매핑 F)
[ 상태 공간: R^{d_model x d_state} (고정된 17 KB 버퍼) ]
```

> **Theorem on Inversion Hardness**  
> 시퀀스 길이 $L > \frac{d_{\mathrm{state}} \cdot d_{\mathrm{model}}}{\log |\mathcal{V}|}$에 대하여, 동일한 상태 $\mathbf{h}_t$를 만들어내는 원본 입력 텍스트의 가짓수(Pre-image Set 크기)는 다음과 같다:
>
> $$|\mathcal{F}^{-1}(\mathbf{h}_t)| \ge |\mathcal{V}|^{L - \frac{d_{\mathrm{model}} d_{\mathrm{state}}}{\log_2 |\mathcal{V}|}} \gg 1$$

- **직관적 설명**:
  - $16\text{,}384$ 단어의 엄청난 정보량을 단 $17\text{ KB}$의 고정된 상태 공간에 욱여넣었기 때문에, **동일한 상태를 만들어내는 가짜 입력 문장이 $50{,}000^{15{,}000}$개 이상 존재**합니다 (비둘기집의 원리).
  - 게다가 최종 출력단에 $\mathbf{y}_t = (\mathbf{h}_t \mathbf{C}_t + \mathbf{D}\tilde{\mathbf{x}}_t) \odot \mathrm{SiLU}(\mathbf{z}_t)$ 라는 고도의 비선형 아다마르 곱(Hadamard Product)이 걸려 있어, 수많은 가짜 극솟값(Spurious Local Minima)이 생겨납니다.
  - 따라서 공격자가 출력값을 아무리 역공학해도 원본 프롬프트를 수학적으로 특정해 내는 것은 **NP-Hard (계산적으로 불가능)**합니다.

---

# 제7장: 정리 5 (보안 정리 2) — 영공간 직교 투영에 의한 완전 프라이버시 소거 (Theorem on Zero-Leakage)

---

### 7.1 정리의 공식 문장 및 수식

> **Theorem (Zero Information Leakage under Arbitrary Non-Linear Probing)**  
> 작동 메모리 상태 $\mathbf{h} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{state}}}$가 비밀 특징 부분공간 $\mathrm{span}(\mathbf{V})$에 속하는 비밀 벡터 $\mathbf{v}_{\mathrm{secret}}$을 포함한다고 하자.  
> 영공간 사영 연산자 $\mathbf{P}_{\perp} = \mathbf{I} - \mathbf{V}^\top (\mathbf{V} \mathbf{V}^\top)^{-1} \mathbf{V}$를 적용한 소거 상태 $\mathbf{h}^* = \mathbf{P}_{\perp} \mathbf{h}$는 다음을 만족한다:
>
> $$\langle \mathbf{v}_{\mathrm{secret}}, \mathbf{h}^* \rangle = \mathbf{v}_{\mathrm{secret}}^\top \left( \mathbf{I} - \mathbf{V}^\top (\mathbf{V}\mathbf{V}^\top)^{-1}\mathbf{V} \right) \mathbf{h} = \mathbf{0}$$
>
> 결과적으로, 임의의 파라미터화된 적대적 심층 비선형 프로브 $\psi_\phi: \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{state}}} \rightarrow \mathcal{Y}$에 대하여, 비밀 라벨 $Y_{\mathrm{secret}}$과 소거된 표현 $\mathbf{h}^*$ 사이의 섀넌 상호정보량(Mutual Information)은 **정확히 $0$**이다:
>
> $$I(Y_{\mathrm{secret}}; \mathbf{h}^*) \equiv 0 \quad \text{and} \quad I(Y_{\mathrm{secret}}; \hat{\mathbf{y}}_t(\mathbf{h}^*)) \equiv 0$$

---

### 7.2 영공간 사영 행렬 $\mathbf{P}_\perp$ 유도 원리

```
1. 비밀 부분공간 기저 행렬: V (K개의 비밀 단어 임베딩을 행으로 쌓음)
2. 비밀 공간으로의 투영 행렬: P_parallel = V^T (V V^T)^{-1} V
3. 비밀 공간을 도려낸 나머지 영공간: P_perp = I - P_parallel
```

- 어떤 벡터 $\mathbf{h}$가 있을 때, 이를 "비밀과 평행한 성분($\mathbf{h}_\parallel$)"과 "비밀과 수직인 성분($\mathbf{h}_\perp$)"으로 쪼갤 수 있습니다 ($\mathbf{h} = \mathbf{h}_\parallel + \mathbf{h}_\perp$).
- $\mathbf{P}_\perp$를 곱하면:
  - $\mathbf{P}_\perp \mathbf{h}_\parallel = (\mathbf{I} - \mathbf{P}_\parallel)\mathbf{h}_\parallel = \mathbf{h}_\parallel - \mathbf{h}_\parallel = \mathbf{0}$ (비밀 성분 완벽 소멸!)
  - $\mathbf{P}_\perp \mathbf{h}_\perp = \mathbf{h}_\perp$ (일반 지식 $100\%$ 보존!)

---

### 7.3 왜 내적 $\langle \mathbf{v}_{\mathrm{secret}}, \mathbf{h}^* \rangle = 0$이 되는가?

수학적으로 직접 전개해 보면 명쾌하게 증명됩니다:

$$\begin{aligned}
\langle \mathbf{v}_{\mathrm{secret}}, \mathbf{h}^* \rangle &= \mathbf{v}_{\mathrm{secret}}^\top \mathbf{P}_\perp \mathbf{h} \\
&= \mathbf{v}_{\mathrm{secret}}^\top \left( \mathbf{I} - \mathbf{V}^\top (\mathbf{V}\mathbf{V}^\top)^{-1}\mathbf{V} \right) \mathbf{h} \\
&= \left( \mathbf{v}_{\mathrm{secret}}^\top - \underbrace{\mathbf{v}_{\mathrm{secret}}^\top \mathbf{V}^\top (\mathbf{V}\mathbf{V}^\top)^{-1}\mathbf{V}}_{=\mathbf{v}_{\mathrm{secret}}^\top \text{ (왜냐하면 } \mathbf{v}_{\mathrm{secret}} \in \mathrm{span}(\mathbf{V}) \text{)} } \right) \mathbf{h} \\
&= \left( \mathbf{v}_{\mathrm{secret}}^\top - \mathbf{v}_{\mathrm{secret}}^\top \right) \mathbf{h} = \mathbf{0} \cdot \mathbf{h} = \mathbf{0}
\end{aligned}$$

---

### 7.4 8계층 심층 딥러닝 프로브로 쥐어짜도 상호정보량 $I \equiv 0$인 이유

- **의문**: "선형 내적이 $0$이라도, 8계층 ResNet이나 MLP 같은 비선형 딥러닝 모델로 비선형 특징을 추출하면 비밀이 새어나오지 않을까?"
- **정보이론적 증명**:
  1. $\mathbf{h}^*$ 안에서 $Y_{\mathrm{secret}}$ 방향의 신호 에너지는 진폭 자체가 $0$입니다.
  2. 따라서 비밀 라벨의 조건부 확률분포는 사전 확률분포와 완벽히 일치합니다: $p(\mathbf{h}^* \mid Y_{\mathrm{secret}} = y) = p(\mathbf{h}^*)$.
  3. 섀넌 상호정보량 정의식에 대입하면:
     $$I(Y_{\mathrm{secret}}; \mathbf{h}^*) = \mathcal{H}(\mathbf{h}^*) - \mathcal{H}(\mathbf{h}^* \mid Y_{\mathrm{secret}}) = \mathcal{H}(\mathbf{h}^*) - \mathcal{H}(\mathbf{h}^*) = 0$$
  4. **데이터 처리 부등식 (Data Processing Inequality)**:
     아무리 강력한 1,000계층 인공신경망 $\psi_\phi$를 후처리로 돌려도, 원본 데이터에 없는 정보량을 새로 만들어낼 수 없습니다 ($I(Y; \psi(\mathbf{h}^*)) \le I(Y; \mathbf{h}^*) = 0$).
  5. 따라서 적대적 프로브의 분류 정확도는 무조건 **$50.0\%$ (동전 던지기 찍기 수준)**로 고정되며, 단 $1\text{ bit}$의 정보도 누출되지 않습니다! $\blacksquare$

---

# 제8장: 총정리 요약 비교표 (Grand Summary Table)

| 비교 항목 | 표준 트랜스포머 (LLaMA-3, GPT-4) | 표준 선형 SSM (Mamba, Mamba-2) | Hokie-LM (NeuroWorld-LM) | 수학적 보증 정리 |
| :--- | :--- | :--- | :--- | :--- |
| **추론 메모리 복잡도** | $\mathcal{O}(T)$ 선형 폭발 (16k에서 32MB/stream) | $\mathcal{O}(1)$ 상수 압축 ($64\text{ KB}$) | **$\mathcal{O}(1) = 17.0\text{ KB}$ 불변 버퍼** | Theorem 1 (Online Regret) |
| **망각 메커니즘** | **불가능** ($\text{Softmax} > 0$, 영구 축적) | 수동적·무차별적 감쇄 ($e^{-\mathbf{A}\Delta t}$) | **CAFE 능동 채널 분할 + 직교 사영 $\mathbf{P}_\perp$** | Theorem 3 (SNR Preservation) |
| **장기 기억 수명 (100k SNR)** | Attention 분산으로 $4.0\text{ dB}$ 붕괴 | 균일 감쇄로 $5.0\text{ dB}$ 기억상실 | **$24.7\text{ dB}$ 무저하 영구 유지** | Theorem 3 (Constant SNR) |
| **100턴 대화 일관성** | $32.0\%$ (30턴 만에 룰 붕괴) | $42.0\%$ (38턴 만에 룰 붕괴) | **$96.2\%$ 일관성 준수** | Theorem 3 (Channel Isolation) |
| **프라이버시/PII 소거** | **불가능** (수억 원 재학습 필요) | **불가능** (주변 문맥까지 함께 파괴) | **$184.3\text{ ms}$ 영공간 투영 ($I \equiv 0$)** | Theorem 5 (Zero-Leakage Unlearning) |
| **심층 추론 방식** | 토큰 낭비형 Verbal CoT ($15\times \sim 350\times$ FLOPs) | 계획 능력 부재 (단순 1스텝 반응형) | **Zero-Token Latent MCTS ($1.25\times$ FLOPs)** | Theorem 2 (Hurwitz Bounded Drift) |
| **단일 H100 서빙 동시성** | 120개 세션에서 OOM (16.38 TB 필요) | $\sim 500$ 세션 | **4,096개 동시 세션 완벽 서빙 (17.41 GB)** | PagedState $\mathcal{O}(1)$ Engine |
