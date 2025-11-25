# Approach Summary
- **Model:** prajjwal1/bert-tiny (2 layers)
- **Data:** Synthetic generation using Faker + Custom STT Noise Injection (lowercase, removed punctuation).
- **Latency Optimization:** - Switched to TinyBERT (<10ms).
  - Applied Dynamic Quantization (int8).
  - Reduced max_sequence_length to 32.
  - Forced single-thread execution (`torch.set_num_threads(1)`).
  
**Results:**
- PII Precision: 96.3%
- p95 Latency: 6.11 ms
