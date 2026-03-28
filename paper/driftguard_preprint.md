# [SECTION 1: INTRODUCTION]

Safety-critical deployment requires inference-time defenses that work on fixed, already-trained language models. Stateless guardrails fail under slow conversational attacks because harmful intent can migrate through a sequence of locally benign turns. DriftGuard addresses this by monitoring hidden-state trajectories online, identifying geometric/topological deviation from a safe manifold before unsafe output appears, and applying causal activation intervention to steer generation back toward safer regions.

Our focus is strictly scale-B drift: representation drift occurring during inference in deployed models, without parameter updates. We target open-source families practical for local iteration and field deployment (Llama-3.1-8B, Mistral-7B class, Gemma 9B-class), and we evaluate robustness under three deployment-relevant shifts: multi-turn intent shaping, quantization differences, and distributional shifts in prompt style.

DriftGuard contributes: (i) a unified drift formalism over token-time hidden trajectories and manifold geometry, (ii) a real-time early-warning score fusing continuity and topology signals, and (iii) a causal intervention module that edits latent states at decode time to reduce downstream unsafe completions.

# [SECTION 2: RELATED WORK AND GAP ANALYSIS]

DeepContext (arXiv:2602.16935) demonstrates that stateful sequence modeling of turn embeddings substantially improves multi-turn jailbreak detection over stateless classifiers, with low runtime overhead. Its core strength is temporal risk accumulation, but it does not directly characterize hidden-state manifold geometry or provide latent-level causal steering.

Bullwinkel et al. (arXiv:2507.02956) show that Crescendo-like jailbreaks can migrate representations toward benign regions turn-by-turn, explaining why single-turn defenses degrade in multi-turn settings. Their representation-engineering analysis clarifies mechanism, but does not propose an online topology-aware early-warning system or closed-loop intervention policy.

Fay et al. (arXiv:2505.20435) and related TDA work identify persistent-homology signatures under adversarial influence, suggesting topology carries robust attack information beyond linear probes. Existing work is mostly offline and analysis-centric; deployment-oriented token-time monitoring and intervention triggers remain underdeveloped.

Additional relevant strands include representation engineering and activation steering (e.g., direction-based control), refusal-direction analyses, and trajectory-level anomaly detection. Across this space, three gaps persist:

1. **Online geometry gap**: most methods analyze static batches, not streaming decode-time trajectories.
2. **Topology-to-action gap**: topological signatures are rarely integrated into practical alarm policies.
3. **Detection-to-intervention gap**: few systems couple early warning with explicit causal latent steering and quantify utility/safety trade-offs.

# [SECTION 3: PROBLEM FORMULATION]

Let a fixed model process a dialogue prefix and produce hidden states at layer $\ell$:
$$
\mathbf{h}_{1:T}^{(\ell)} = \left(\mathbf{h}_1^{(\ell)}, \ldots, \mathbf{h}_T^{(\ell)}\right), \quad \mathbf{h}_t^{(\ell)} \in \mathbb{R}^d.
$$

We monitor continuity and compactness/topology under a sliding window $\mathcal{W}_t$.

Continuity metric:
$$
\cos(\theta_t) = \frac{\mathbf{h}_t^\top \mathbf{h}_{t-1}}{\|\mathbf{h}_t\|_2\|\mathbf{h}_{t-1}\|_2}.
$$

Local Lipschitz proxy:
$$
L_t = \frac{\|\mathbf{h}_t-\mathbf{h}_{t-1}\|_2}{\|\mathbf{h}_{t-1}\|_2 + \epsilon}.
$$

Compactness/topology uses reduced points $\tilde{\mathbf{h}}_i$ (PCA-projected):
$$
\mathrm{diam}(\mathcal{W}_t) = \max_{i,j \in \mathcal{W}_t}\|\tilde{\mathbf{h}}_i-\tilde{\mathbf{h}}_j\|_2,
$$
and persistent homology summaries $\beta_k(\mathcal{W}_t)$ and persistence mass.

We define fused risk:
$$
R_t = \alpha \cdot \phi\!\left(1-\cos(\theta_t)\right) + (1-\alpha)\cdot \psi(L_t) + \gamma \cdot \xi(\Delta \beta_k, \Delta \mathrm{diam}),
$$
with alarm $A_t=\mathbb{1}[R_t \ge \tau]$.

# [SECTION 4: DRIFTGUARD PIPELINE]

**Early-warning module.** At each token step, DriftGuard computes continuity features and periodic topology snapshots over recent hidden states. Alarming is calibrated on benign traces to bound false alarms while maximizing lead time before unsafe emission.

**Causal intervention module.** On alarm, hidden states are steered toward a safe reference manifold by bounded interpolation:
$$
\mathbf{h}_t' = (1-\lambda)\mathbf{h}_t + \lambda \mathbf{h}_{\text{safe}}, \quad \lambda \in [0,1].
$$
The induced logit correction is applied for current-token decoding and bounded by a norm budget.

**Operational objective.** Maximize unsafe-output prevention and alarm lead time while minimizing utility loss and latency overhead.

# [SECTION 5: EXPERIMENTAL DESIGN]

We evaluate three threat classes: (i) gradual multi-turn jailbreak chains (Crescendo-style), (ii) deception/sycophancy drift prompts, and (iii) emergent misalignment under shifted prompt distributions and quantization changes.

Primary model set: Llama-3.1-8B and Gemma-class 8B/9B-equivalent models for local iterations; scale-out replication on larger hardware where feasible.

Metrics:
- Detection AUROC and precision-recall.
- Lead time: tokens between first alarm and first unsafe completion.
- Intervention efficacy: unsafe completion rate reduction.
- Utility cost: refusal overfire, benign-task degradation.
- Runtime overhead per generated token.

Calibration and testing are separated by conversation family to avoid leakage in trajectory shape.

# [SECTION 6: EXPECTED VISUALS]

**Figure 1 (Trajectory PCA map).** Caption: "Token-time latent trajectories under benign vs Crescendo-style prompts." The figure shows benign paths concentrated in a compact basin, while adversarial multi-turn paths drift gradually and cross alarm contours before unsafe completion.

**Figure 2 (Persistence diagrams/barcodes).** Caption: "Topological signatures before and after intent drift." The figure highlights increased long-lived features and lobe emergence in high-risk windows.

**Figure 3 (Cosine/Lipschitz heatmap).** Caption: "Continuity degradation over token-time." Rows are sessions, columns are token steps, color encodes fused risk.

**Figure 4 (Lead-time distribution).** Caption: "Early-warning lead time across attack families." The plot reports median and interquartile lead time, comparing continuity-only vs continuity+topology alarms.

# [SECTION 7: LIMITATIONS AND SAFETY NOTES]

Persistent homology remains computationally expensive at long contexts, requiring subsampling and dimensionality reduction. Thresholds may shift across model families and quantization settings, requiring per-deployment calibration. Steering can trade off harmful-response suppression with over-refusal in ambiguous benign queries; bounded intervention budgets and abstain policies are necessary.

# [SECTION 8: APPENDIX A — CODE OUTLINES]

```python
# nnsight-style pseudocode for layer capture
with model.trace(prompt) as tracer:
    h = model.model.layers[layer_idx].output[0].save()
```

```python
# TransformerLens-style pseudocode for activation steering
def steering_hook(resid, hook):
    return resid + alpha * safe_direction
logits = model.run_with_hooks(tokens, fwd_hooks=[("blocks.20.hook_resid_post", steering_hook)])
```

```python
# Drift score loop outline
for t in decode_steps:
    h_t = hidden_state(layer_idx, t)
    cos_t = cosine(h_t, h_prev)
    L_t = lipschitz_proxy(h_t, h_prev)
    topo_t = topology_snapshot(window)
    R_t = fuse(cos_t, L_t, topo_t)
    if R_t >= tau:
        apply_steering()
```

# [SECTION 9: REPRODUCIBILITY CHECKLIST]

- Fixed seeds for data split and decoding.
- Explicit model revision and tokenizer version logging.
- Saved per-token traces and alarm decisions.
- Exported scripts for figure regeneration.
- Hardware profile recorded (local MPS vs multi-GPU CUDA).

# [SECTION 10: TARGET BIBLIOGRAPHY]

1. DeepContext: Stateful Real-Time Detection of Multi-Turn Adversarial Intent Drift (arXiv:2602.16935).
2. Bullwinkel et al., A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks (arXiv:2507.02956).
3. Fay et al., The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology (arXiv:2505.20435).
4. Russinovich et al., Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack (arXiv:2404.01833).
5. Zou et al., Representation Engineering: A Top-Down Approach to AI Transparency (arXiv:2310.01405).
6. Turner et al., Steering Language Models With Activation Engineering (arXiv:2308.10248).
7. Farquhar et al., Detecting Hallucinations in Large Language Models Using Semantic Entropy (Nature 2024).
8. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs (arXiv:2406.15927).
9. TransformerLens documentation and library release (tooling reference for intervention hooks).
10. nnsight tracing/intervention documentation (tooling reference for causal activation edits).
