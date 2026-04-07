# Prompt Engineering for Open-Weight Models

Practical lessons from testing prompt variants on open-weight LLMs for multi-axis ideological scoring.
For exact numbers and statistical analysis, see [PAPER.pdf](../PAPER.pdf) (arXiv submission pending).

---

## The Problem

Large proprietary models (GPT-class) can handle complex multi-axis scoring tasks out of the box: you give them a long prompt with 8 ideological axes, a JSON schema, and a political proposal, and they return stable, well-calibrated scores.

Open-weight models (tested here with `llama3.1:8b`) struggle significantly with the same task. Common failure modes include:

- **Axis confusion**: the model conflates conceptually adjacent axes (e.g., scoring "equality vs. liberty" identically to "state vs. market")
- **Score clustering**: outputs collapse to a narrow range (e.g., all scores between -0.2 and 0.2) regardless of proposal content
- **Language interference**: when the prompt is in English but the proposal is in another language, the model loses sensitivity to the proposal's actual content
- **Schema violations**: malformed JSON, missing axes, or scores outside the valid range

The question is whether prompt engineering alone can close this gap, without fine-tuning.

---

## Prompt Variants Tested

Six prompt variants were evaluated against the same set of Spanish political proposals, using `llama3.1:8b` via Ollama. Each variant was compared to GPT-class baselines on inter-rater reliability and score distribution quality.

### 1. Baseline

The same English-language prompt used with GPT, applied to the open-weight model with no modifications. This serves as the control condition.

### 2. Language-Matched

The full prompt is translated into the language of the proposals (Spanish in this case). Axis names, scale anchors, and all instructions are provided in the same language the model will read in the proposal text.

### 3. Chain-of-Thought (CoT)

An instruction to "think step by step, analyzing each axis before producing the JSON output" is prepended to the scoring request. The model is expected to reason through each axis explicitly before emitting the structured output.

### 4. Few-Shot

Three calibration examples are included in the prompt: one proposal that should score strongly positive on most axes, one strongly negative, and one near-neutral. Each example includes the expected JSON output.

### 5. Task Decomposition

Instead of scoring all 8 axes in a single call, each axis is scored in a separate API call with a focused, single-axis prompt. The results are then assembled into the full score vector.

### 6. Language-Matched + Simplified

Combines language matching with prompt simplification: shorter axis descriptions, reduced preamble, and removal of nuanced edge-case instructions. The hypothesis is that smaller models benefit from both linguistic alignment and reduced cognitive load.

---

## Summary of Findings

| Variant | Effect on Score Quality | Practical Impact | Cost |
|---------|------------------------|------------------|------|
| Baseline | Poor reliability, high axis confusion | Not recommended for production use | 1x |
| Language-matched | Strongest single improvement | **Most effective** -- recommended as default | 1x |
| Chain-of-thought | Modest improvement in score spread | Helpful but not sufficient on its own | ~1.3x (longer output) |
| Few-shot | Degraded performance | **Counterproductive** -- models mimic example tone | ~1.5x (longer prompt) |
| Task decomposition | Improved per-axis stability | Effective but expensive | 8x |
| Language-matched + simplified | Strong improvement, comparable to language-matched alone | **Recommended starting point** for resource-constrained setups | 1x |

---

## Key Findings in Detail

### Language matching is the most impactful single change

Translating the prompt into the proposal language produced the largest improvement of any variant tested. This held even when the model's training data is predominantly English. The intuition is straightforward: when the model processes the prompt and proposal in the same language, it maintains a more coherent internal representation of the task.

This effect was most pronounced on axes where the score depends on subtle phrasing in the proposal text. English-prompted models tended to assign bland, centrist scores to proposals whose ideological valence was clear to a native reader.

### Few-shot examples backfire

This was the most counterintuitive finding. Including calibration examples in the prompt -- a standard technique for improving LLM task performance -- actually degraded scoring quality with `llama3.1:8b`.

The failure mode is specific: rather than using the examples as scale anchors (understanding what a +0.8 or -0.6 "looks like"), the model pattern-matched on surface features of the example proposals. If the positive example used formal language, the model would score other formally-written proposals more positively regardless of their ideological content.

This suggests that smaller models lack the capacity to simultaneously hold calibration examples in context and apply them abstractly to new inputs. Practitioners should test few-shot carefully before assuming it helps.

### Chain-of-thought helps modestly

Adding explicit reasoning instructions improved score spread (less clustering around zero) and slightly improved axis differentiation. However, the improvement was smaller than language matching alone, and combining CoT with language matching did not produce substantial additional gains.

CoT also increases output token count, which adds latency and cost in local inference setups.

### Task decomposition improves stability at high cost

Scoring one axis at a time reduces the cognitive load on the model and produces more stable per-axis scores. However, it requires 8 separate API calls per proposal, making it roughly 8 times more expensive in compute. For large-scale labeling, this is often impractical.

Task decomposition is most useful as a diagnostic tool: if decomposed scores are substantially better than joint scores, it confirms that the model is struggling with multi-axis reasoning rather than with the axes themselves.

---

## Recommendations for Practitioners

1. **Always match prompt language to proposal language.** This is the single highest-leverage change and costs nothing extra. Even if you are more comfortable writing prompts in English, translate them.

2. **Start with the simplified + language-matched variant.** For models in the 7-8B parameter range, shorter prompts with clear, direct instructions outperform verbose prompts with nuanced guidance.

3. **Do not assume few-shot examples will help.** Test with and without examples on a held-out set before committing to few-shot prompting. Smaller models are particularly prone to surface-level pattern matching on examples.

4. **Use task decomposition for validation, not production.** Run a small batch with decomposed scoring to diagnose whether your model can handle the multi-axis task. If decomposed scores are much better, consider a larger model rather than paying the 8x cost permanently.

5. **Measure reliability, not just accuracy.** A model that gives reasonable-looking scores on a single run may still have poor test-retest reliability. Always score a subset of proposals multiple times and compute agreement metrics.

6. **Consider model scale.** The findings here are specific to `llama3.1:8b`. Larger open-weight models (70B+) may not need the same level of prompt adaptation. Test before generalizing.

---

## Scope and Limitations

These findings come from testing on `llama3.1:8b` with Spanish-language political proposals scored on 8 ideological axes. Several open questions remain:

- **Other languages**: the language-matching effect likely generalizes, but the magnitude may differ for languages with more or less LLM training data.
- **Other models**: newer open-weight model families (Mistral, Qwen, Gemma) may respond differently to these prompt variants. The few-shot backfire effect in particular may not replicate on larger or more capable models.
- **Other task structures**: these results apply to multi-axis Likert-style scoring. Binary classification or ranking tasks may have different sensitivity to prompt engineering choices.

Contributions testing these variants on other models and languages are welcome -- see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## Related Documentation

- [Axis definitions](axes.md) -- the 8 ideological axes used in scoring
- [Methodology](methodology.md) -- the full scoring and validation pipeline
- [Prompt template](../prompts/label_8axis_v1.txt) -- the baseline prompt used for GPT and as the starting point for variants
