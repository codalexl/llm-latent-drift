# [SECTION 1: INTRODUCTION]

Inference-time drift in hidden representations is an under-instrumented failure channel for safety-critical LLM deployment. In practical deployments, models are frozen and cannot be retrained on demand; therefore, safety must be enforced by runtime monitoring and intervention over internal states rather than parameter updates. We focus on **scale-B drift**: trajectory-level representational migration during autoregressive generation under realistic perturbations (multi-turn conversational pressure, prompt-distribution shift, and quantization).

DriftGuard is an online monitor-controller for fixed deployed models. It tracks token-time hidden trajectories, computes continuity and topological compactness signals, triggers early warnings before unsafe completions, and applies bounded latent steering toward calibrated safe reference states. The central hypothesis is that unsafe conversational migration is detectable as geometric drift before policy-breaking text manifests, and that causal steering at alarm time can reduce downstream unsafe behavior with acceptable utility and latency costs.

The operational target is not post-hoc explanation but **real-time prevention**. Our design requirement is therefore end-to-end: detection and intervention must share a unified metric family, remain efficient at decode time, and produce deployment-auditable telemetry (alarm lead time, intervention frequency, and per-token overhead).

# [SECTION 2: LITERATURE REVIEW AND GAP ANALYSIS]

### 2.1 Multi-turn attack dynamics and stateful detection
DeepContext (arXiv:2602.16935) demonstrates that stateful temporal modeling of turn embeddings improves multi-turn adversarial detection versus stateless guardrails. Bullwinkel et al. (arXiv:2507.02956) further show, from a representation engineering perspective, that Crescendo-style attacks can migrate hidden states through benign-looking regions, exposing why single-turn detectors fail in chained interactions. Crescendo (arXiv:2404.01833) establishes the threat realism: gradual, low-salience conversational escalation can reliably bypass one-shot defenses.

**Limitation:** these works establish temporal migration and detection value, but do not provide a topology-aware token-time alarm function coupled to causal latent intervention in one online loop.

### 2.2 Geometric/topological interpretability under adversarial pressure
Fay et al. (arXiv:2505.20435) characterize adversarial influence via persistent homology, showing robust topological signatures across model families. This supports the view that manifold structure, not only linear directions, carries attack information.

**Limitation:** most TDA results are offline and analysis-first. Translation to streaming decode-time policies is limited, especially under strict latency budgets.

### 2.3 Activation-level control and representation engineering
RepE (arXiv:2310.01405) formalizes population-level directions for reading and controlling high-level attributes. Activation engineering (arXiv:2308.10248) and conditional activation steering (arXiv:2409.05907) show inference-time controllability without finetuning, including selective refusal behavior.

**Limitation:** steering is often evaluated as unconditional attribute control or static benchmark manipulation, not as a closed-loop response to online drift alarms with calibrated early-warning guarantees.

### 2.4 Gap synthesis
Across these lines, four deployment-critical gaps remain:

1. **Streaming geometry gap:** sparse adoption of token-time geometry/topology signals for online inference loops.
2. **Alarm consistency gap:** detector metrics and runtime intervention triggers are often mismatched.
3. **Topology-to-control gap:** topological signatures rarely gate causal intervention policies.
4. **Safety-utility accounting gap:** few systems jointly report lead time, unsafe-output reduction, benign over-refusal, and latency overhead.

DriftGuard is designed to close these four gaps with a single, reproducible runtime pipeline.

# [SECTION 3: FORMAL PROBLEM STATEMENT]

Let a fixed autoregressive model produce hidden states at monitored layer $\ell$:

$$
\mathbf{h}_{1:T}^{(\ell)} = \left(\mathbf{h}_1^{(\ell)}, \dots, \mathbf{h}_T^{(\ell)}\right), \quad \mathbf{h}_t^{(\ell)} \in \mathbb{R}^d.
$$

For each decoding step $t$, define continuity and local sensitivity:

$$
\cos(\theta_t)=\frac{\mathbf{h}_t \cdot \mathbf{h}_{t-1}}{\lVert \mathbf{h}_t \rVert_2 \lVert \mathbf{h}_{t-1} \rVert_2}, \qquad
L_t=\frac{\lVert \mathbf{h}_t-\mathbf{h}_{t-1} \rVert_2}{\lVert \mathbf{h}_{t-1} \rVert_2+\epsilon}.
$$

Over a sliding window $\mathcal{W}_t=\{\mathbf{h}_{t-w+1},\dots,\mathbf{h}_t\}$, project to $\tilde{\mathbf{h}}$ with PCA for efficient TDA and define compactness:

$$
\mathrm{diam}(\mathcal{W}_t)=\max_{i,j \in \mathcal{W}_t}\lVert \tilde{\mathbf{h}}_i-\tilde{\mathbf{h}}_j \rVert_2.
$$

Using persistent homology on the reduced point cloud, compute Betti summaries $\beta_0(\mathcal{W}_t), \beta_1(\mathcal{W}_t)$ and persistence mass $\Pi_t$.

We use normalized risk terms:

$$
r_t^{\mathrm{cont}} = \max\!\left(0,\frac{\tau_{\cos}-\cos(\theta_t)}{1-\tau_{\cos}}\right), \quad
r_t^{\mathrm{lip}}  = \max\!\left(0,\frac{L_t-\tau_L}{\tau_L}\right),
$$

$$
r_t^{\mathrm{topo}}=
\max\!\left(0,\frac{\mathrm{diam}(\mathcal{W}_t)-\tau_{\mathrm{diam}}}{\tau_{\mathrm{diam}}}\right)+
\max\!\left(0,\frac{\beta_1(\mathcal{W}_t)-\tau_{\beta_1}}{\tau_{\beta_1}}\right).
$$

The fused online risk is:

$$
R_t = w_c r_t^{\mathrm{cont}} + w_l r_t^{\mathrm{lip}} + w_t r_t^{\mathrm{topo}}, \qquad
A_t=\mathbb{1}[R_t \ge \tau_R].
$$

On alarm, we apply bounded latent correction:

$$
\mathbf{h}_t'=(1-\lambda)\mathbf{h}_t+\lambda\mathbf{h}_{\mathrm{safe}}, \quad \lambda \in [0,1],
$$

with optional norm budget $\lVert \mathbf{h}_t'-\mathbf{h}_t \rVert_2 \le \delta$ and cache-consistent decoding after intervention.

# [SECTION 4: DRIFTGUARD ONLINE PIPELINE]

### 4.1 Monitoring loop
At each token:
1. capture layer hidden state $\mathbf{h}_t$,
2. update continuity metrics $(\cos(\theta_t), L_t)$,
3. update topology snapshot every $s$ steps over window size $w$,
4. compute fused risk $R_t$ and alarm $A_t$.

Outputs are per-step logs: token id, continuity/topology metrics, risk score, alarm flag, steering flag, and latency.

### 4.2 Causal intervention
If $A_t=1$, DriftGuard computes a safe-reference pull and adjusts the current decoding distribution via steered hidden logits. To preserve autoregressive consistency, the runtime invalidates stale KV state after steering and resumes generation from recomputed full-prefix context.

### 4.3 Calibration and deployment mode
Thresholds $(\tau_{\cos}, \tau_L, \tau_{\mathrm{diam}}, \tau_{\beta_1}, \tau_R)$ are calibrated on benign traces for each deployed model/quantization profile. DriftGuard supports standard HF decoding and nnsight tracing variants, both at inference time only.

# [SECTION 5: EXPERIMENTAL DESIGN]

### 5.1 Models and deployment settings
Primary local-GPU iteration targets: Llama-3.1-8B, Mistral-7B-Instruct, Gemma-2-9B (plus Gemma-3/Qwen variants for ablations). Scale-up replications can include 27B/70B-class models where hardware permits. All runs are strictly no-finetune inference-time evaluations.

### 5.2 Threat suites
We evaluate three threat families:
- **Gradual multi-turn jailbreak drift:** Crescendo-like conversational escalation.
- **Deception/sycophancy drift:** prompts that pressure agreement, concealment, or manipulative compliance.
- **Quantization shift:** identical prompt sets under bf16/fp16 versus 4-bit deployment.

### 5.3 Metrics
Detection:
- AUROC and PR-AUC from session-level risk maxima,
- precision/recall/F1 from alarm decisions.

Early warning:
- first-alarm lead time (tokens before session endpoint or unsafe event proxy),
- boundary crossing frequency and risk smoothness (offline summaries).

Intervention and utility:
- unsafe output rate (with optional judge labels),
- benign over-refusal / benign compliance retention,
- intervention rate and steered-step fraction.

Systems:
- mean per-token latency and steering-induced overhead.

### 5.4 Reproducibility protocol
All experiment entry points expose explicit random seeds, fixed preset suites, and machine-readable JSON outputs. Figures are generated by scripts from stored outputs rather than manual notebook edits.

# [SECTION 6: RESULTS TEMPLATE AND ANALYSIS PLAN]

### 6.1 Expected quantitative pattern
We expect fused continuity+topology alarms to improve early-warning recall under multi-turn migration versus continuity-only variants, especially where adversarial paths remain locally smooth but globally deform manifold structure. Quantization should shift calibration and reduce AUROC without recalibration, motivating per-profile thresholding.

### 6.2 Causal steering efficacy
We test whether alarm-gated steering decreases unsafe outputs on unsafe prompts while controlling benign degradation. The key reportable quantity is:

$$
\Delta_{\mathrm{unsafe}} = \Pr(\text{unsafe output}\mid\text{steer}) - \Pr(\text{unsafe output}\mid\text{no steer}),
$$

with analogous benign compliance delta:

$$
\Delta_{\mathrm{benign}} = \Pr(\text{compliant benign output}\mid\text{steer}) - \Pr(\text{compliant benign output}\mid\text{no steer}).
$$

### 6.3 Safety interpretation
A practically useful detector-controller should satisfy:
1. positive lead time on unsafe sessions,
2. reduced unsafe-output rate after steering,
3. bounded false-alarm and benign compliance loss,
4. acceptable decode overhead for production latency budgets.

# [SECTION 7: FIGURES AND CAPTIONS]

**Figure 1 — Token-time PCA Trajectory Drift.**  
Caption: *Safe and unsafe token trajectories projected onto principal latent axes. Unsafe multi-turn sessions exhibit progressive manifold displacement relative to benign basins.*

**Figure 2 — Persistence Diagram under Drift.**  
Caption: *Persistent homology of unsafe trajectory windows reveals increased long-lived $H_1$ features, indicating lobe formation and topological deformation under adversarial pressure.*

**Figure 3 — Cosine Drift Heatmap.**  
Caption: *Heatmap of $1-\cos(\theta_t)$ over sessions and token positions. High-drift bands concentrate before alarm onset in unsafe trajectories.*

**Figure 4 — Lead-Time Distribution.**  
Caption: *Histogram/violin of first-alarm lead time on unsafe prompts. Positive mass away from zero indicates actionable early warning before endpoint failure.*

**Figure 5 — Runtime Overhead Bar Plot.**  
Caption: *Mean per-token latency for no-steer versus steer conditions; quantifies systems cost of intervention.*

# [SECTION 8: SELF-CRITIQUE, LIMITATIONS, AND FAILURE MODES]

1. **TDA scaling:** persistent homology cost grows with context length; practical operation requires windowing and dimensionality reduction, potentially missing long-range structure.
2. **Threshold transportability:** calibrated thresholds are model/quantization dependent and may not transfer across architectures without recalibration.
3. **Proxy mismatch risk:** if unsafe-output labels rely on weak judges or endpoint proxies, intervention benefit may be misestimated.
4. **Steering side effects:** latent correction can increase benign refusals in ambiguous contexts; this requires constrained steering strength and abstention policies.
5. **Adaptive adversaries:** attackers may optimize around known alarm surfaces; robust deployment requires randomized checks and periodic red-team refresh.

# [SECTION 9: APPENDIX A — NNsight IMPLEMENTATION OUTLINE]

```python
from nnsight import LanguageModel
import torch

model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
layer_idx = 20

def capture_last_hidden(prompt: str):
    with model.trace(prompt):
        h_proxy = model.model.layers[layer_idx].output[0].save()
        logits_proxy = model.lm_head.output.save()
    h = h_proxy.value  # [1, T, d]
    logits = logits_proxy.value[:, -1, :]
    return h[:, -1, :], logits

def steer_once(prompt: str, safe_ref: torch.Tensor, alpha: float):
    with model.trace(prompt):
        layer_out = model.model.layers[layer_idx].output[0]
        delta = alpha * (safe_ref.to(layer_out.device) - layer_out[:, -1, :])
        layer_out[:, -1, :] = layer_out[:, -1, :] + delta
        logits = model.lm_head.output.save().value[:, -1, :]
    return logits
```

# [SECTION 10: APPENDIX B — TRANSFORMERLENS OUTLINE]

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
hook_name = "blocks.20.hook_resid_post"

def steering_hook(resid, hook, safe_ref, alpha):
    # resid: [batch, pos, d]
    delta = alpha * (safe_ref.to(resid.device) - resid[:, -1, :])
    resid[:, -1, :] = resid[:, -1, :] + delta
    return resid

def run_with_steering(tokens, safe_ref, alpha=0.2):
    hooks = [(hook_name, lambda r, h: steering_hook(r, h, safe_ref, alpha))]
    return model.run_with_hooks(tokens, fwd_hooks=hooks)
```

# [SECTION 11: APPENDIX C — METRIC COMPUTATION OUTLINE]

```python
import numpy as np
from latent_dynamics.tda_metrics import topology_snapshot

def drift_loop(hidden_seq, cfg):
    prev = None
    window = []
    risk = []
    for h in hidden_seq:
        cos = None if prev is None else float(np.dot(h, prev) / (np.linalg.norm(h) * np.linalg.norm(prev) + 1e-8))
        lip = None if prev is None else float(np.linalg.norm(h - prev) / (np.linalg.norm(prev) + 1e-8))
        window.append(h)
        topo = None
        if len(window) >= cfg.topology_window:
            topo = topology_snapshot(np.asarray(window[-cfg.topology_window:], dtype=np.float32))
        r = fuse(cos, lip, topo, cfg)  # weighted normalized terms
        risk.append(r)
        prev = h
    return np.asarray(risk, dtype=np.float32)
```

# [SECTION 12: TARGET BIBLIOGRAPHY]

1. DeepContext: Stateful Real-Time Detection of Multi-Turn Adversarial Intent Drift (arXiv:2602.16935).
2. Bullwinkel et al., A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks (arXiv:2507.02956).
3. Fay et al., The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology (arXiv:2505.20435).
4. Russinovich et al., Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack (arXiv:2404.01833).
5. Zou et al., Representation Engineering: A Top-Down Approach to AI Transparency (arXiv:2310.01405).
6. Turner et al., Steering Language Models With Activation Engineering (arXiv:2308.10248).
7. Rimsky et al., Programming Refusal with Conditional Activation Steering (arXiv:2409.05907).
8. TransformerLens documentation (activation patching and intervention hooks).
9. nnsight documentation (trace-time hidden capture and causal intervention).
10. Latent-representation jailbreak detection studies (e.g., arXiv:2602.11495) for comparative baselines.
