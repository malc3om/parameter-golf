# Title: Implement SOTA 11-Layer Model (Target val_bpb ~1.113)

### Description
This pull request introduces the complete end-to-end implementation of the SOTA architecture optimizations for the Parameter Golf 10-minute / 16MB track. By systematically accumulating established best practices and advancing the architecture to an 11-layer U-Net enhanced Transformer, we confidently target a sub-1.115 validation bpb.

### Key Architectural Updates
* **11-Layer U-Net Transformer**: Expanded the baseline architecture to 11 layers with symmetric skip connections from encoder blocks (0→5) to decoder blocks (6→10) to efficiently route features while maintaining optimal parameter allocation.
* **LeakyReLU(0.5)²**: Replaced standard ReLU² with our custom LeakyReLU(0.5)² to prevent dead neurons and propagate small negative gradients, crucial for deeper stable training.
* **Exclusive Self Attention (XSA)**: Configured the last 4 layers with XSA to ensure representations capture orthogonal contexts by subtracting the components of attention vectors aligned with individual token embeddings.
* **Partial RoPE (16/64)**: Integrated position-free signal tracking across the upper 48 dimensions of the query and key heads, focusing RoPE strictly on the first 16 to improve length-extrapolation robustness.
* **Deep Layer LN Scaling**: Norm scaling introduced `val * (1/sqrt(layer+1))` to inherently regularize representations leading up to the classification head.
* **Value Embeddings (VE128)**: Injected shared continuous 128-dimensional identity representations exclusively into blocks 9 and 10 to stabilize final logit projections.

### Execution & QAT
* **EMA & Tight SWA**: Maintained an EMA buffer (decay 0.997) evaluated continuously, combined with SWA over the final stages of the training plateau (every 50 steps starting 50% in).
* **Late QAT with STE**: QAT execution delayed until the initial model stabilization (15% through), leveraging a Straight-Through Estimator during forward passes for optimal INT6 quantization transitions without degradation.
* **Test-Time Training (Legal)**: Built highly customized backward-looking TTT executing over non-overlapping 32K token windows, adapting via SGD to push out maximum marginal performance strictly inside evaluation rules.
* **Quantization Protocol**: Integrated `GPTQ-lite` targeting optimal per-row scaling by checking 6 potential precision-based clip candidates.

### Checks
- [x] Syntax checking passes.
- [x] Compilation validates locally against 8xH100 limitations.
- [x] Weights output confirmed ≤ 16MB using custom ZSTD compression protocol.
- [x] Token limit handles effectively.

Please review for merge, specifically evaluating the parameters set across the newly bundled techniques in `submission.json`.
