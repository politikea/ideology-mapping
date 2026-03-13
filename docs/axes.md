# The 8 Ideological Axes

This document explains each axis, its design rationale, and what a score means in practice.
The definitive operational specification is `prompts/label_8axis_v1.txt` — that is the prompt the AI model reads.
This document provides additional context and examples.

---

## Score Interpretation

All scores are continuous floats from −100 to +100.

| Range | Interpretation |
|-------|---------------|
| −100 to −80 | Dominant orientation toward the negative pole |
| −80 to −50 | Strong preference for the negative pole |
| −50 to −20 | Moderate lean toward the negative pole |
| −20 to +20 | Balanced, ambiguous, or mixed — the proposal does not take a clear side |
| +20 to +50 | Moderate lean toward the positive pole |
| +50 to +80 | Strong preference for the positive pole |
| +80 to +100 | Dominant orientation toward the positive pole |

A score near 0 is not a failure — it means the proposal does not take a clear ideological stance on this axis. Many proposals are genuinely ambiguous on some axes.

---

## The 8 Axes

### aequitas_libertas (Equity vs. Freedom)

**−100 pole**: Full redistribution; equality of outcome; tax-funded provision of goods and services.  
**+100 pole**: Individual economic freedom; minimal redistribution; market outcomes respected as legitimate.

**Examples**:
- "Implement a universal basic income of €900/month" → strong negative (−70): pure redistribution
- "Cut public spending by 30%" → moderate positive (+40): less state intervention
- "Improve public transport routes" → near 0 (+5): provision without redistribution framing

**Design rationale**: This is the classic economic left-right axis. It is the single most informative axis in the corpus (PC1 loading = −0.493 in Phase 1 PCA). It is highly correlated with `market_collective_allocation` (r = −0.92), which is why the compression leaderboard's top pairs often include both.

---

### imperium_anarkhia (Authority vs. Decentralization)

**−100 pole**: Centralized state authority; uniform national policy; strong central government.  
**+100 pole**: Decentralization; local autonomy; self-organization; subsidiarity.

**Examples**:
- "Increase police forces by 20%" → strongly negative (−35): central authority enforcement
- "Eliminate public party funding" → moderate positive (+25): less central state control
- "Implement a national water plan" → moderately negative (−20): centralized management

**Design rationale**: This is the axis Politikea uses as one of the two user-facing dimensions. It captures state-structure preferences that are orthogonal to economic left-right. A left-wing decentralist and a right-wing libertarian can both score positive here for different reasons — the axis is not ideologically pure by design.

---

### universalism_particularism (Universal vs. Group-Specific)

**−100 pole**: Universal rules and rights; one law for all; context-blind policy.  
**+100 pole**: Group-specific policies; contextual exceptions; differential treatment by identity, region, or circumstance.

**Examples**:
- "Expropriate empty bank-owned housing for social rental" → moderate negative (−10): applies universally to a class of properties
- "Create special economic zones for disadvantaged regions" → positive (+35): differentiated treatment by location

**Design rationale**: This axis has the lowest VIF (1.29) — it is the most independent axis in the space. It captures a dimension that does not correlate strongly with the economic cluster. In Phase 1, `Identidad y Cohesión Social` proposals cluster strongly positive here (+35.3).

---

### market_collective_allocation (Market vs. Collective)

**−100 pole**: Market allocation of goods and services; price signals; private provision.  
**+100 pole**: Collective or democratic allocation; public provision; planned or cooperative economics.

**Examples**:
- "Privatize hospital management" → strongly negative (−40): market allocation of healthcare
- "Create a public housing agency" → positive (+55): collective allocation of housing
- "Regulate pharmacy opening hours" → moderate positive (+20): democratic constraints on market

**Design rationale**: Highly correlated with `aequitas_libertas` (r = −0.92). The distinction is between *who gets what* (aequitas: about equality of outcome) vs. *how it's allocated* (market: about mechanism). In practice they measure nearly the same thing.

---

### inequality_acceptance_correction (Natural vs. Structural Inequality)

**−100 pole**: Inequality is natural or acceptable; unequal outcomes reflect merit or circumstance.  
**+100 pole**: Inequality is structural; must be actively corrected by policy.

**Examples**:
- "Cut inheritance tax" → negative (−30): allows wealth to pass without correction
- "Implement mandatory wage transparency" → positive (+40): structural correction

**Design rationale**: Different from `aequitas_libertas` in that it captures the *diagnosis* (is inequality a problem?) rather than the *prescription* (what to do about it). In practice highly correlated with the economic cluster.

---

### individual_socialized_risk (Individual vs. Pooled Risk)

**−100 pole**: Individuals bear their own risk; no cross-subsidization.  
**+100 pole**: Risk pooled collectively; insurance is social.

**Examples**:
- "Eliminate private pension funds in favor of public system" → strongly positive (+75): full risk socialization
- "Introduce individual healthcare accounts" → strongly negative (−60): individual risk-bearing

**Design rationale**: Related to but distinct from `market_collective_allocation`. Focuses specifically on risk pooling — insurance logic — rather than production or provision. Relevant for healthcare, pensions, unemployment.

---

### progressivism_preservation (Change vs. Continuity)

**−100 pole**: Active reform; embrace change; update institutions and culture.  
**+100 pole**: Stability; tradition; continuity; caution about change.

**Examples**:
- "Legalize cannabis use" → strongly negative (−45): reform of law and social norms
- "Maintain current retirement age" → positive (+25): preserve existing system
- "Teach financial literacy in secondary school" → moderately negative (−20): reform of curriculum

**Design rationale**: One of the two PC2 components (loading = +0.674). Captures a cultural-temporal dimension orthogonal to the economic cluster. In `Seguridad y Defensa` proposals, this axis shows a surprising pattern: positive (conservative/preservation-oriented) proposals co-occur with strongly anti-authority (`imperium_anarkhia`) scores.

---

### technocracy_populism (Expert-Led vs. People-Led)

**−100 pole**: Expert-led governance; technocratic decision-making; deference to specialists.  
**+100 pole**: Popular-will primacy; participatory democracy; people over experts.

**Examples**:
- "Establish an independent scientific advisory body for climate policy" → negative (−15): expert governance
- "Mandatory binding referendums on major budget decisions" → positive (+30): popular will
- "Reduce bureaucracy" → moderate positive (+15): less technocratic overhead

**Design rationale**: Second-lowest VIF (1.38) — nearly orthogonal to the economic cluster. This axis is the second-most independent axis in the space. In Phase 1, `Identidad y Cohesión Social` proposals also score high here (+27.9), combining universalism with technocratic framing.

---

## Which Axes Are Most Independent?

| Axis | VIF | Interpretation |
|------|-----|----------------|
| universalism_particularism | 1.29 | Nearly orthogonal — carries unique signal |
| technocracy_populism | 1.38 | Nearly orthogonal — carries unique signal |
| progressivism_preservation | 1.84 | Mostly independent |
| imperium_anarkhia | 2.50 | Partially correlated with economic cluster |
| inequality_acceptance_correction | 6.88 | High collinearity with economic cluster |
| individual_socialized_risk | 8.20 | High collinearity with economic cluster |
| market_collective_allocation | 9.03 | Very high collinearity with economic cluster |
| aequitas_libertas | 10.44 | Highest collinearity — redundant with economic cluster |

The four economic cluster axes (bottom four) are nearly interchangeable for capturing economic left-right. `universalism_particularism` and `technocracy_populism` are where the 8D framework adds the most information beyond a simpler economic axis.
