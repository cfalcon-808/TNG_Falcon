# AI Disclosure Materials

This folder contains the AI-use records for the VAE portion of the project.

## Included Chats

| Thread | Raw JSONL | Main contribution |
| --- | --- | --- |
| Plan VAE dataset pipeline | `raw/rollout-2026-05-05T16-41-46-019dfb29-adce-7fd1-8f8c-4dbd376c207a.jsonl` | Planning the VAE dataset structure, labels, and generator workflow |
| Guide dataset generator | `raw/rollout-2026-05-06T10-58-36-019dff15-dc8f-7bb1-b443-f8fcf4dae046.jsonl` | Guidance on completing the dataset generator and selecting checkpoint strengths |
| Locate VAE training data | `raw/rollout-2026-05-06T15-33-04-019e0011-259a-78b0-8665-a45e158665b2.jsonl` | Locating and organizing VAE training datasets and artifacts |
| Use VAE for better start states | `raw/rollout-2026-05-08T10-16-56-019e093c-7262-77d2-a53a-00365993c45f.jsonl` | Discussing VAE value use, phase gating, and reward-shaping integration |

## Folder Layout

| Path | Purpose |
| --- | --- |
| `raw/` | Original Codex JSONL exports |
| `readable/` | Markdown transcripts generated from the raw logs |
| `Convert-CodexJsonlToMarkdown.ps1` | Local converter used to generate the readable transcripts |

## Regenerate Readable Transcripts

```powershell
powershell -ExecutionPolicy Bypass -File AI_DISCLOSURE/Convert-CodexJsonlToMarkdown.ps1
```
