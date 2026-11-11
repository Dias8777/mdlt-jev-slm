| system                                       | all 3 right   | category   | urgency   | escalate   | invalid   |   latency, s | cost per 1,000 decisions                  | auto-decided at ≥95 % accuracy   |
|:---------------------------------------------|:--------------|:-----------|:----------|:-----------|:----------|-------------:|:------------------------------------------|:---------------------------------|
| Jev 1.14 (4 typed questions), API            | 93.8 %        | 98.8 %     | 95.0 %    | 100.0 %    | 0.0 %     |         0.44 | $0.0389                                   | 94 %                             |
| DeepSeek V4 Flash (JSON, T = 0), API         | 93.8 %        | 98.8 %     | 96.2 %    | 98.8 %     | 0.0 %     |         1.22 | $0.0362                                   | 0 % *                            |
| DeepSeek V4 Flash (JSON, T = 1), API         | 88.8 %        | 98.8 %     | 90.0 %    | 100.0 %    | 0.0 %     |         1.2  | $0.0362                                   | 91 %                             |
| DeepSeek V4 Pro (JSON, T = 0), API           | 87.5 %        | 97.5 %     | 93.8 %    | 96.2 %     | 0.0 %     |         1.45 | $0.0754                                   | 0 % *                            |
| Qwen3-0.6B + LoRA, PyTorch on Apple M4 (mps) | 77.5 %        | 93.8 %     | 86.2 %    | 93.8 %     | 0.0 %     |         1.02 | $0 on own laptop (≈ $0.099 on a cloud T4) | 39 %                             |
| Qwen3-0.6B + LoRA, GGUF q8_0 in LM Studio    | 78.8 %        | 93.8 %     | 86.2 %    | 93.8 %     | 0.0 %     |         1.42 | $0 on own laptop                          | —                                |
| Qwen3-0.6B base (no fine-tuning)             | 12.5 %        | 56.2 %     | 36.2 %    | 51.2 %     | 0.0 %     |         1.02 | $0 on own laptop                          | 0 %                              |

\* no usable confidence: every answer has probability 1.0, so the system can only decide all messages or none, and all are decided automatically only if accuracy is ≥ 95 %.
