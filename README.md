# TurkishSTS

This repository contains code I wrote whilst building a Turkish Semantic Textual Similarity models. I don't really feel
like explaining the entire procedure because there were many failed attempts. However the final setup combined [SimCSE++](https://arxiv.org/abs/2305.13192)
style unsupervised training with a regular [SimCSE](https://arxiv.org/abs/2104.08821) supervised training.

To view the model and for more details, please visit [this](https://huggingface.co/DolphyWind/bert-base-turkish-cased-simcse) link.

## Evaluation

Scores are correlations between cosine similarity and gold scores, x100. Best result per column in bold.

|Dataset|Split|
|---|---|
|STS-B (`emrecan/stsb-mt-turkish`)|validation|
|NLI-TR (`emrecan/all-nli-tr`, pair-score)|test|
|CxC (`DolphyWind/cxc-sts-mt-turkish`, 80-20-split)|validation|
|SICK-R (`DolphyWind/sickr-sts-mt-turkish`)|test|
|STR-2022 (machine-translated)|train|

### Spearman (cosine)

|Model|STS-B|NLI-TR|CxC|SICK-R|STR-2022|Avg|
|---|---|---|---|---|---|---|
|**DolphyWind/bert-base-turkish-cased-simcse**|83.92|49.88|67.25|**76.07**|**79.82**|**71.39**|
|emrecan/bert-base-turkish-cased-mean-nli-stsb-tr|**86.05**|47.43|66.95|75.55|77.06|70.61|
|ytu-ce-cosmos/modernbert-tr-embed|80.24|48.72|**67.82**|75.21|75.65|69.53|
|trmteb/turkish-embedding-model-fine-tuned|81.71|**50.76**|65.74|73.93|71.65|68.76|
|ytu-ce-cosmos/turkish-e5-large|83.19|39.02|66.85|75.44|76.00|68.10|

### Pearson (cosine)

|Model|STS-B|NLI-TR|CxC|SICK-R|STR-2022|Avg|
|---|---|---|---|---|---|---|
|**DolphyWind/bert-base-turkish-cased-simcse**|84.31|**50.23**|**69.10**|**80.77**|**80.23**|**72.93**|
|emrecan/bert-base-turkish-cased-mean-nli-stsb-tr|**86.05**|47.66|68.83|80.70|77.83|72.21|
|ytu-ce-cosmos/modernbert-tr-embed|79.67|47.98|68.86|79.98|76.52|70.60|
|trmteb/turkish-embedding-model-fine-tuned|81.01|50.20|67.08|78.69|71.83|69.76|
|ytu-ce-cosmos/turkish-e5-large|82.98|39.61|68.35|79.48|77.47|69.58|

