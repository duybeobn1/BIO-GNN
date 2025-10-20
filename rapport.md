
## Link Prediction on the Plane Network

## 1. Dataset Description

The dataset represents an **airport network**, where each node corresponds to a city or airport characterized by several features:

- Longitude (`lon`)
- Latitude (`lat`)
- Population
- Country (categorical, one-hot encoded)
- City name (string attribute)

After preprocessing, we obtained **3363 nodes** and **27,094 undirected edges**. Each node is described by a **215-dimensional feature vector** combining numerical and categorical attributes. The task is **link prediction**: predicting whether a connection (route) exists between two airports using graph-based methods.

***

## 2. Models and Baselines

We compared **learning-based methods** (GAE and VGAE) against **classical topological heuristics** commonly used for link prediction.

### 2.1 Learning-based models

- **Graph Autoencoder (GAE)**:  
  A deterministic encoder-decoder model using two GCN layers for node embedding and a dot-product decoder for edge reconstruction.

- **Variational Graph Autoencoder (VGAE)**:  
  Similar to GAE but introduces Gaussian latent variables (`μ`, `logσ`) and a Kullback-Leibler (KL) divergence term for probabilistic embedding regularization.

### 2.2 Classical heuristics

- **Jaccard Coefficient**:  
  Measures the ratio of common neighbors to total neighbors between two nodes.

  $$
  J(u,v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}
  $$

- **Preferential Attachment (PA)**:  
  Models hub tendency, based on the product of node degrees.

  $$
  PA(u,v) = |N(u)| \times |N(v)|
  $$

These baseline methods rely purely on graph connectivity and require no training or node features.

***

## 3. Features sets analysis

In order to verify which dataset allows for the best performance, we first performed an ablation study on the Features themselves.

### 3.1 - Feature sets definition

We decomposed our original node features into 3 distinctive sets : 
- Full features : all the data available for each airport (each graph node) : population, lat, lon, country.
- Numerical features only : a subset of our original features, without the country name.
- No node features : removal of all features for all nodes.

#### No mode features

In order to simulate the "no node feature" mode, we entered an identity matrix defined as `np.eye(df.shape[0], dtype=np.float32)` as node features in our models. This is equivalent to representing each node by a **unique one-hot vector** with no other information.

The idea behind this "empty" feature set is to remove all semantic information to our graph. This forces the model to learn embeddings based on **connectivity patterns alone**, hence from pure **graph structure**. This baseline will help us understand the impact of our features on model performance.

### 3.2 - Experimental setup 

This study was performed using the following setup : 

- **Data split:** 80% train edges, 10% test edges, 10% validation edges (using `train_test_split_edges` from PyTorch Geometric)
- **Latent dimension:** 16
- **Optimizer:** Adam
- **Learning rates tested:** 0.01, 0.001 (best result selected)
- **Epochs:** 600
- **Evaluation metrics:** Area Under ROC Curve (AUC) and Average Precision (AP)

Both AUC and AP were computed on the test link set (`test_pos_edge_index`, `test_neg_edge_index`).

### 3.3 - Results

| **Feature Set** | **Model** | **AUC** | **Average Precision (AP)** |
|------------------|-----------|----------|-----------------------------|
| **Full features** | Linear | 0.8840 | 0.8787 |
|  | VGAE | 0.9220 | 0.9119 |
|  | GAE | **0.9461** | **0.9439** |
|  | Jaccard (classical) | 0.9346 | 0.9307 |
|  | Preferential Attachment (classical) | 0.9101 | 0.9213 |
| **Numerical features only** | Linear | 0.8302 | 0.8193 |
|  | VGAE | 0.8869 | 0.8746 |
|  | GAE | **0.9117** | **0.9089** |
|  | Jaccard (classical) | 0.9346 | 0.9307 |
|  | Preferential Attachment (classical) | 0.9101 | 0.9213 |
| **No node features** | Linear | 0.9195 | 0.9357 |
|  | VGAE | 0.8702 | 0.8820 |
|  | GAE | 0.9188 | 0.9328 |
|  | Jaccard (classical) | **0.9346** | **0.9307** |
|  | Preferential Attachment (classical) | 0.9101 | 0.9213 |

***

## 4. Models comparison

### 4.1 - Experimental setup 

The setup of this experiment is very similar to the one previously used :

- **Data split:** different variations compared
- **Latent dimension:** 16
- **Optimizer:** Adam
- **Learning rates tested:** 0.01, 0.001 (best result selected)
- **Epochs:** 600
- **Evaluation metrics:** Area Under ROC Curve (AUC) and Average Precision (AP)

Both AUC and AP were computed on the test link set (`test_pos_edge_index`, `test_neg_edge_index`).

### 4.2 - Results


#### Summary Table

| **Split Configuration** | **Model** | **AUC** | **AP** | **Train %** |
|------------------------|-----------|---------|--------|-------------|
| **10% Test / 10% Val** | Linear | 0.8840 | 0.8787 | 80% |
|                        | VGAE | 0.9220 | 0.9119 | 80% |
|                        | GAE | **0.9461** | **0.9439** | 80% |
|                        | Jaccard | 0.9346 | 0.9307 | - |
|                        | PA | 0.9101 | 0.9213 | - |
| **20% Test / 10% Val** | Linear | 0.8847 | 0.8814 | 70% |
|                        | VGAE | 0.9219 | 0.9151 | 70% |
|                        | GAE | **0.9480** | **0.9443** | 70% |
|                        | Jaccard | 0.9369 | 0.9340 | - |
|                        | PA | 0.9073 | 0.9189 | - |
| **20% Test / 20% Val** | Linear | 0.8834 | 0.8777 | 60% |
|                        | VGAE | 0.9147 | 0.9087 | 60% |
|                        | GAE | 0.9367 | 0.9383 | 60% |
|                        | Jaccard | 0.9173 | 0.9137 | - |
|                        | PA | 0.9125 | 0.9214 | - |
| **30% Test / 10% Val** | Linear | 0.8817 | 0.8770 | 60% |
|                        | VGAE | 0.9126 | 0.9027 | 60% |
|                        | GAE | 0.9346 | 0.9330 | 60% |
|                        | Jaccard | 0.9360 | 0.9326 | - |
|                        | PA | 0.9094 | 0.9201 | - |

#### Summary

**Best Split**: The **20% test / 10% validation** split (70% training) achieves the best overall performance across all models, with GAE reaching the highest AUC of 0.9480 and AP of 0.9443.

**Key Findings**:
- **Training data matters**: Models with 70-80% training data significantly outperform those with only 60% training data. Reducing training data from 80% to 60% causes GAE to drop from AUC 0.9461 to 0.9346-0.9367.
- **GAE consistently dominates**: GAE outperforms VGAE across all splits, confirming that deterministic embeddings work better for this clean, structured airport network.
- **VGAE degrades faster**: VGAE shows higher sensitivity to reduced training data, dropping from AUC 0.9220 (80% train) to 0.9126 (60% train).
- **Classical heuristics remain stable**: Jaccard and PA maintain relatively consistent performance (AUC 0.91-0.93) regardless of split ratio since they don't require training.

***

## 5 - Experiments Analysis 

### 5.1 - Feature sets analysis

### a. Full Features
- **GAE (AUC 0.9461, AP 0.9439)** performs best, showing that deterministic embeddings from GCNs with full node attributes excel.
- **VGAE (AUC 0.9220, AP 0.9119)** lags slightly behind GAE, likely due to the KL regularization that adds noise.
- **Linear encoder (AUC 0.8840, AP 0.8787)** is weaker but surprisingly strong, demonstrating that some linear feature transformations capture useful info.
- **Classical heuristics (Jaccard AUC 0.9346, PA AUC 0.9101)** do well too, but GAE improves further by incorporating node features.

#### b. Numerical Features Only
- All scores drop compared to full features, indicating **categorical features (e.g., country encoding) add predictive value**.
- GAE (AUC 0.9117) outperforms classical PA (AUC 0.9101) , but VGAE underpeforms classical 
- Linear drops more steeply (AUC 0.8302), showing the benefit of nonlinear modeling for numeric-only features.

#### c. No Node Features (Identity matrix)
- Linear encoder surprisingly performs very well (AUC 0.9195, AP 0.9357), showing that unique node identifiers plus simple linear maps can still encode connectivity well.
- GAE (AUC 0.9188, AP 0.9328) also performs strongly, close to heuristics.
- VGAE slightly worse (AUC 0.8702); stochastic sampling may not help here.
- Classical heuristics remain very competitive.

#### Overall Insights
- We can validate that our **node features enhance prediction:** full features > numerical only > none.
- However, we remark that **topology alone (no features) remains very powerful**.
- **Linear encoder is a strong baseline**, particularly with unique node features (identity matrix). It can surprisingly rival nonlinear VGAE, highlighting the importance of choosing strong baselines to properly evalute the relevance of a model.

### 5.2 - Overall models comparison

#### a. **GAE Consistently Outperforms VGAE**

**Observation:**
- GAE outperforms VGAE in **all feature settings** (full, numerical, none).
- With full features: GAE AUC 0.9461 vs VGAE 0.9220 (difference of ~0.024).
- With numerical only: GAE 0.9117 vs VGAE 0.8869 (~0.025 difference).
- With no features: GAE 0.9188 vs VGAE 0.8702 (~0.049 difference).

**Our analysis**
- VGAE introduces a **KL divergence regularization** term that enforces the latent space to match a prior Gaussian distribution.
- According to our research, this regularization is beneficial for **sparce, noisy or incomplete data**, but your airport network is clean, deterministic, and highly structured.
- As a result, in our case, the noise added by stochasticity in VGAE **degrades the predictions** instead of improving generalization.

#### b. **Classical Heuristics Are Remarkably Strong**

**Observation:**
- Jaccard coefficient consistently achieves **AUC 0.9346 and AP 0.9307** regardless of feature configuration.
- Preferential Attachment (PA) achieves **AUC 0.9101 and AP 0.9213**, also stable.

**Our analysis**
- Both heuristics rely **purely on graph topology**: neighborhood overlap (Jaccard) and degree product (PA).
- This is why they remain unchaged, whatever the model used. They are **featureless baselines** because they don't use node attributes at all.
- Since the airport network connectivity is highly **topological** (airports connect based on existing routes and hub structures), these heuristics naturally capture the dominant link formation mechanism.
- Thanks to the strong baseline they offer for link prediction, we know that any learned model must surpass ~0.93 AUC to justify its complexity.
- Still, we managed to outperform these heuristics can be surpassed using rich features and GCN embeddings.

***

### c. **Full Features Enable Best Performance**

**Observation:**
- GAE with full features achieves the **highest AUC (0.9461)** and AP (0.9439).
- VGAE with full features also performs well (AUC 0.9220).
- Linear encoder improves from 0.8302 (numerical only) to 0.8840 (full features).

**Why?**
- Full features include **categorical encoding (country one-hot)**, which adds relational context not present in pure numerical features.
- Country encoding helps the model learn **regional connectivity patterns** (e.g., airports in the same country or region are more likely connected).
- GCN layers aggregate these rich features across neighborhoods, enabling more **expressive latent representations**.

**Implication:**
- **Categorical features matter**: adding country information boosts performance across all models.
- This shows that **domain-specific features** (region, category) can significantly enhance learned embeddings.

***

### 4. **Numerical Features Alone Are Weaker**

**Observation:**
- Dropping categorical features (keeping only population, lat, lon) **decreases all model scores**.
- GAE drops from 0.9461 to 0.9117 (~3.4% drop).
- VGAE drops from 0.9220 to 0.8869 (~3.5% drop).
- Linear drops from 0.8840 to 0.8302 (~5.4% drop).

**Why?**
- Numerical features (lat, lon, population) are continuous and spatially informative but lack **discrete relational context**.
- Geographic coordinates might help slightly (nearby airports connect), but without categorical grouping (country), the model cannot easily learn regional clusters.
- Linear models especially struggle with continuous features alone since they lack the expressiveness to capture nonlinear geographic patterns.

**Implication:**
- **Categorical encoding** is crucial for capturing **cluster-level patterns** in geographically distributed networks.
- Pure continuous features are less informative without structural or categorical context.

***

### 5. **No Features: Surprising Strength of Topology**

**Observation:**
- With **identity matrix** (no features), GAE achieves AUC 0.9188, very close to Jaccard (0.9346).
- Linear encoder with identity features achieves **AUC 0.9195**, AP 0.9357 — surprisingly the **highest AP** among no-feature models.
- VGAE performs worst at 0.8702.

**Why?**
- Identity matrix provides **unique node identifiers** with no semantic information.
- GAE and Linear models learn embeddings purely from **graph connectivity** via message passing or unique indexing.
- This shows the **graph structure itself is highly predictive**, and models can still learn meaningful embeddings from topology alone.
- Linear encoder's strong performance suggests that **one-hot node IDs** + simple transformations can encode positional and topological information effectively [3].

**Implication:**
- **Topology dominates** link prediction in this dataset.
- Node features are **helpful but not essential** — the network's connectivity patterns already encode sufficient information.
- This aligns with classical heuristics' success: they rely purely on topology and perform comparably.

***

### 6. **Linear Encoder: Underrated Baseline**

**Observation:**
- Linear encoder achieves competitive scores:
  - Full features: AUC 0.8840
  - No features (identity): AUC 0.9195, **AP 0.9357** (highest among no-feature models)

**Why?**
- With identity features, linear transformations can learn **position-specific embeddings** effectively.
- This simple model acts as a **strong baseline** showing what non-graph-based learning can achieve.
- It highlights that **model complexity** (GCN layers, variational sampling) must be justified by significant performance gains.

**Implication:**
- Always include **simple baselines** (linear, logistic regression) in ablation studies.
- If a linear model performs nearly as well, it questions the necessity of complex architectures.

***

### 7. **Variational Models Struggle with Deterministic Data**

**Observation:**
- VGAE consistently underperforms GAE across all feature settings.
- The gap widens with fewer features (no features: GAE 0.9188 vs VGAE 0.8702).

**Why?**
- VGAE's variational framework assumes **latent uncertainty**, modeling embeddings as distributions rather than points.
- This is useful when data is **incomplete, noisy, or probabilistic**.
- However, your airport network has **clean, deterministic edges** (routes either exist or don't), so the added stochasticity introduces noise without benefit.

**Implication:**
- Use **VGAE for uncertain or probabilistic graphs** (e.g., social networks with ambiguous connections, missing data).
- For **deterministic graphs**, stick with deterministic models like GAE.

***

## What This Means for Your Study

### Main Conclusions:

1. **Topology is king**: Classical heuristics and no-feature models perform near-optimally, showing link formation is driven by connectivity patterns.

2. **Node features add value when rich**: Full features (including categorical) push GAE to the highest performance, but numerical features alone are insufficient.

3. **Deterministic models > Variational models**: For clean, structured graphs, GAE consistently outperforms VGAE.

4. **Simple baselines matter**: Linear encoders with identity features rival complex GNNs, emphasizing the need for justified complexity.

5. **Categorical features are critical**: Country encoding significantly boosts all models, highlighting the importance of domain-specific feature engineering.

