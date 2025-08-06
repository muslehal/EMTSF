# EMTSF

Abstract. The immense success of the Transformer architecture
in Natural Language Processing has led to its adoption in Time Series Forecasting (TSF), where superior performance has been shown.
However, a recent important paper questioned their effectiveness by
demonstrating that a simple single layer linear model outperforms
Transformer-based models. This was soon shown to be not as valid,
by a better transformer-based model termed PatchTST. More recently, TimeLLM demonstrated even better results by reprogramming i.e., repurposing a Large Language Model (LLM) for the TSF
domain. Again, a follow up paper challenged this by demonstrating
that removing the LLM component or replacing it with a basic attention layer in fact yields better performance. One of the challenges in
forecasting is the fact that TSF data favors the more recent past, and is
sometimes subject to unpredictable events. Based upon these recent
insights in TSF, we propose a Mixture of Experts (MoE) framework.
Our method combines the state-of-the-art (SOTA) models including
xLSTM, enhanced Linear, PatchTST, minGRU among others. This
set of complimentary and diverse models for TSF are integrated in a
Transformer MoE model. Our results on standard TSF benchmarks
demonstrate better results surpassing all current TSF models, including those based on recent MoE frameworks.
