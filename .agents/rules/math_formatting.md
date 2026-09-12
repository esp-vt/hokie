# Mathematical Expression Formatting Rule

## Mandatory Constraint:
When generating responses, explanations, markdown files, and documentation:
1. **Unconditional Backtick Wrapping**:
   - ALL mathematical variables, symbols, constants, and formulas MUST be wrapped in backticks (`` `like this` ``).
   - Examples:
     - Variables/Constants: `h_t`, `z_t`, `γ_t`, `d_model`, `17.0 KB`, `Ω`, `ε`, `ρ(A)`, `P_⊥`
     - Operations & Inner Products: `u^T v`, `⟨u, v⟩`, `u v^T`, `A · h_{t-1}`
     - Complexity & Information Bounds: `O(1)`, `O(T)`, `R_T ≤ O(M · d_model · log T)`, `I(X; Y) ≡ 0`
     - Metrics & Ratios: `SNR = 24.7 dB`, `184.3 ms`, `1.25x FLOPs`, `96.2%`
2. **Clean Unicode Math over Raw LaTeX**:
   - Do NOT use raw LaTeX macros like `\mathcal{H}`, `\mathbf{u}`, `\mathbb{R}`, `\frac{a}{b}`, `\text{...}` in text/markdown.
   - Use clean, human-readable Unicode math symbols inside backticks:
     - `H(X)` instead of `\mathcal{H}(X)`
     - `D_KL(q || p)` instead of `\mathcal{D}_{\mathrm{KL}}(q \parallel p)`
     - `u^T v` or `⟨u, v⟩` instead of `\mathbf{u}^\top \mathbf{v}`
     - `P_⊥` instead of `\mathbf{P}_\perp`
     - `≤`, `≥`, `≠`, `≡`, `∈`, `→`, `∞`, `Σ`, `√`, `|| · ||`, `∂`, `Δ`, `Ω`, `γ`, `θ`, `φ`, `ε`, `ρ`, `σ`, `⊥`
3. **Display Equations / Proofs**:
   - Multi-line equations and proofs must be wrapped in markdown code blocks or blockquotes with backticks for each line.
