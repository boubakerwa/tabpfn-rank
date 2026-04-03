# Phase 2 Pointwise Decision Memo

TabPFN version: `v2.5`
Native adapter outcome: **keep as targeted cold-start variant**
Pointwise TabPFN story holds: **yes**

## Reasons

- tabpfn_native had positive mean NDCG@10 deltas on 2/4 key MovieLens slices.
- Item-cold bootstrap support was positive on at least one slice for yes >=2/3 seeds.
- Median native-vs-ohe runtime ratio across key slices was 0.782.
- Best TabPFN variant was within 1% of the best tree on 4/4 key slices.
- K=50 retention on item_cold / global_popularity with tabpfn_native was 1.0061.
- K=50 retention on item_cold / context_popularity with tabpfn was 0.9630.

## Native vs One-Hot Key Slices

- warm / global_popularity: delta=0.0112, runtime_ratio=0.7915
- warm / context_popularity: delta=-0.0305, runtime_ratio=0.7928
- item_cold / global_popularity: delta=0.0163, runtime_ratio=0.7671
- item_cold / context_popularity: delta=-0.0022, runtime_ratio=0.7724

## Bootstrap Support

- warm / global_popularity / 10% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=-0.0053, CI=[-0.0304, 0.0194]
- warm / global_popularity / 10% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.1179, CI=[0.0762, 0.1572]
- warm / global_popularity / 10% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=-0.0053, CI=[-0.0287, 0.0188]
- warm / global_popularity / 10% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.1179, CI=[0.0786, 0.1569]
- warm / global_popularity / 10% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=-0.0053, CI=[-0.0297, 0.0181]
- warm / global_popularity / 10% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.1179, CI=[0.0781, 0.1581]
- warm / global_popularity / 20% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=-0.0066, CI=[-0.0235, 0.0099]
- warm / global_popularity / 20% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0145, CI=[-0.0082, 0.0406]
- warm / global_popularity / 20% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=-0.0066, CI=[-0.0237, 0.0090]
- warm / global_popularity / 20% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0145, CI=[-0.0100, 0.0416]
- warm / global_popularity / 20% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=-0.0066, CI=[-0.0259, 0.0092]
- warm / global_popularity / 20% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.0145, CI=[-0.0103, 0.0411]
- warm / global_popularity / 50% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=0.0198, CI=[-0.0017, 0.0428]
- warm / global_popularity / 50% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0793, CI=[0.0407, 0.1238]
- warm / global_popularity / 50% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0198, CI=[0.0001, 0.0425]
- warm / global_popularity / 50% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0793, CI=[0.0408, 0.1211]
- warm / global_popularity / 50% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=0.0198, CI=[-0.0008, 0.0437]
- warm / global_popularity / 50% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.0793, CI=[0.0417, 0.1204]
- warm / global_popularity / 100% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=0.0112, CI=[-0.0029, 0.0289]
- warm / global_popularity / 100% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0460, CI=[0.0127, 0.0834]
- warm / global_popularity / 100% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0112, CI=[-0.0044, 0.0298]
- warm / global_popularity / 100% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0460, CI=[0.0131, 0.0821]
- warm / global_popularity / 100% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=0.0112, CI=[-0.0035, 0.0295]
- warm / global_popularity / 100% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.0460, CI=[0.0126, 0.0832]
- item_cold / global_popularity / 10% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=0.0417, CI=[0.0111, 0.0744]
- item_cold / global_popularity / 10% / best_tabpfn_minus_best_tree / seed 0: mean_delta=-0.0456, CI=[-0.0883, -0.0018]
- item_cold / global_popularity / 10% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0273, CI=[-0.0040, 0.0602]
- item_cold / global_popularity / 10% / best_tabpfn_minus_best_tree / seed 1: mean_delta=-0.1280, CI=[-0.1701, -0.0814]
- item_cold / global_popularity / 10% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=0.1207, CI=[0.0874, 0.1554]
- item_cold / global_popularity / 10% / best_tabpfn_minus_best_tree / seed 2: mean_delta=-0.1499, CI=[-0.2009, -0.0994]
- item_cold / global_popularity / 20% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=0.0314, CI=[0.0040, 0.0608]
- item_cold / global_popularity / 20% / best_tabpfn_minus_best_tree / seed 0: mean_delta=-0.0405, CI=[-0.0723, -0.0080]
- item_cold / global_popularity / 20% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0611, CI=[0.0317, 0.0937]
- item_cold / global_popularity / 20% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0337, CI=[0.0030, 0.0662]
- item_cold / global_popularity / 20% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=0.0181, CI=[0.0037, 0.0388]
- item_cold / global_popularity / 20% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.0673, CI=[0.0368, 0.1012]
- item_cold / global_popularity / 50% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=0.0044, CI=[-0.0103, 0.0197]
- item_cold / global_popularity / 50% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0199, CI=[-0.0060, 0.0489]
- item_cold / global_popularity / 50% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0570, CI=[0.0275, 0.0900]
- item_cold / global_popularity / 50% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0436, CI=[0.0124, 0.0774]
- item_cold / global_popularity / 50% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=0.0379, CI=[0.0200, 0.0599]
- item_cold / global_popularity / 50% / best_tabpfn_minus_best_tree / seed 2: mean_delta=-0.0026, CI=[-0.0221, 0.0173]
- item_cold / global_popularity / 100% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=0.0235, CI=[0.0050, 0.0443]
- item_cold / global_popularity / 100% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0130, CI=[-0.0035, 0.0317]
- item_cold / global_popularity / 100% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0223, CI=[0.0038, 0.0437]
- item_cold / global_popularity / 100% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0135, CI=[-0.0002, 0.0320]
- item_cold / global_popularity / 100% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=0.0030, CI=[-0.0021, 0.0111]
- item_cold / global_popularity / 100% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.0071, CI=[-0.0107, 0.0292]
- item_cold / context_popularity / 10% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=0.0037, CI=[-0.0141, 0.0218]
- item_cold / context_popularity / 10% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0471, CI=[0.0162, 0.0807]
- item_cold / context_popularity / 10% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0121, CI=[-0.0005, 0.0253]
- item_cold / context_popularity / 10% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0974, CI=[0.0583, 0.1396]
- item_cold / context_popularity / 10% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=-0.1469, CI=[-0.1891, -0.1030]
- item_cold / context_popularity / 10% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.0238, CI=[-0.0159, 0.0656]
- item_cold / context_popularity / 20% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=-0.0839, CI=[-0.1183, -0.0520]
- item_cold / context_popularity / 20% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0537, CI=[0.0258, 0.0866]
- item_cold / context_popularity / 20% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0399, CI=[0.0179, 0.0653]
- item_cold / context_popularity / 20% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0425, CI=[0.0196, 0.0688]
- item_cold / context_popularity / 20% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=-0.0309, CI=[-0.0590, -0.0068]
- item_cold / context_popularity / 20% / best_tabpfn_minus_best_tree / seed 2: mean_delta=-0.0210, CI=[-0.0505, 0.0048]
- item_cold / context_popularity / 50% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=0.0043, CI=[-0.0323, 0.0397]
- item_cold / context_popularity / 50% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0256, CI=[-0.0172, 0.0707]
- item_cold / context_popularity / 50% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=-0.0392, CI=[-0.0727, -0.0078]
- item_cold / context_popularity / 50% / best_tabpfn_minus_best_tree / seed 1: mean_delta=-0.0319, CI=[-0.0800, 0.0114]
- item_cold / context_popularity / 50% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=-0.0347, CI=[-0.0615, -0.0113]
- item_cold / context_popularity / 50% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.0672, CI=[0.0358, 0.1011]
- item_cold / context_popularity / 100% / tabpfn_native_minus_tabpfn / seed 0: mean_delta=-0.0445, CI=[-0.0757, -0.0198]
- item_cold / context_popularity / 100% / best_tabpfn_minus_best_tree / seed 0: mean_delta=0.0729, CI=[0.0417, 0.1083]
- item_cold / context_popularity / 100% / tabpfn_native_minus_tabpfn / seed 1: mean_delta=0.0092, CI=[-0.0127, 0.0323]
- item_cold / context_popularity / 100% / best_tabpfn_minus_best_tree / seed 1: mean_delta=0.0044, CI=[-0.0381, 0.0485]
- item_cold / context_popularity / 100% / tabpfn_native_minus_tabpfn / seed 2: mean_delta=0.0288, CI=[-0.0053, 0.0669]
- item_cold / context_popularity / 100% / best_tabpfn_minus_best_tree / seed 2: mean_delta=0.0860, CI=[0.0489, 0.1248]

## K Sensitivity

- Tree comparisons in this section use the tree model selected from the primary 100% train sweep for each split/protocol slice.
- item_cold / global_popularity / tabpfn_native: K50 retention=1.0061
- item_cold / context_popularity / tabpfn: K50 retention=0.9630

## Amazon Sanity

- Amazon is secondary evidence here and was run with capped query counts for directional sanity checking.
- Amazon warm / global_popularity was not saturated.
- Amazon warm / context_popularity best model: tabpfn_native (NDCG@10=0.5697).
- Amazon item_cold / global_popularity was saturated.
- Amazon item_cold / context_popularity best model: tabpfn_native (NDCG@10=0.9380).

## Feature Group Ablation

- context_popularity / tabpfn / full: NDCG@10=0.9087
- context_popularity / tabpfn / metadata_only: NDCG@10=0.5456
- context_popularity / tabpfn / no_interaction_history: NDCG@10=0.5456
- context_popularity / tabpfn / no_item_metadata: NDCG@10=0.8698
- context_popularity / tabpfn / no_user_metadata: NDCG@10=0.9087
- context_popularity / tabpfn_native / full: NDCG@10=0.8642
- context_popularity / tabpfn_native / metadata_only: NDCG@10=0.5519
- context_popularity / tabpfn_native / no_interaction_history: NDCG@10=0.5519
- context_popularity / tabpfn_native / no_item_metadata: NDCG@10=0.8698
- context_popularity / tabpfn_native / no_user_metadata: NDCG@10=0.8642
- global_popularity / tabpfn / full: NDCG@10=0.9548
- global_popularity / tabpfn / metadata_only: NDCG@10=0.6263
- global_popularity / tabpfn / no_interaction_history: NDCG@10=0.6263
- global_popularity / tabpfn / no_item_metadata: NDCG@10=0.9889
- global_popularity / tabpfn / no_user_metadata: NDCG@10=0.9548
- global_popularity / tabpfn_native / full: NDCG@10=0.9782
- global_popularity / tabpfn_native / metadata_only: NDCG@10=0.6003
- global_popularity / tabpfn_native / no_interaction_history: NDCG@10=0.6003
- global_popularity / tabpfn_native / no_item_metadata: NDCG@10=0.9889
- global_popularity / tabpfn_native / no_user_metadata: NDCG@10=0.9782
