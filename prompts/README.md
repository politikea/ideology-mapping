# Prompts

This directory contains the labeling prompt and its structured output schema.

## Files

| File | Purpose |
|------|---------|
| `label_8axis_v1.txt` | The scoring prompt — substitute `{{TEXT}}` with a political proposal |
| `label_8axis_v1_schema.json` | JSON Schema for the expected response (for API structured output enforcement) |

## How to Use

### 1. Prepare the prompt

Replace `{{TEXT}}` with the full text of a political proposal:

```python
prompt_template = open("prompts/label_8axis_v1.txt").read()
prompt = prompt_template.replace("{{TEXT}}", proposal_text)
```

### 2. Call the model

Use any frontier model (Claude, GPT-4o, or equivalent). Call it **N = 10–13 times per proposal** at `temperature = 0.2`. The multi-run design is essential — it provides ICC reliability estimates and sign agreement per axis.

```python
import anthropic, json

client = anthropic.Anthropic()  # or use OpenAI SDK with base_url for other providers

responses = []
for _ in range(13):
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    responses.append(json.loads(resp.content[0].text))
```

### 3. Structured output (optional)

If your API supports structured output enforcement, pass `label_8axis_v1_schema.json` as the response schema. This guarantees valid JSON with all required fields.

### 4. Expected response

Each call returns a JSON object with:
- **8 axis scores** (`axis_*`): float, -100 to +100
- **8 confidence scores** (`conf_axis_*`): float, 0 to 1
- **`global_confidence`**: float, 0 to 1
- **`flags`**: array of strings (empty if no special conditions)
- **`rationale_spans`**: array of character-offset spans pointing to evidence in the input text

### Recommended settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.2 | Low enough for reliability, high enough for ICC estimation |
| Runs per proposal | 10–13 | Enough for stable ICC(2,1) estimates |
| Max tokens | 1024 | Response is typically ~400 tokens |
| Response format | `json_object` or structured output | Prevents malformed responses |
