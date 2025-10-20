
## Link Prediction on the Plane Network

### 1. Dataset Description

The dataset represents an **airport network**, where each node corresponds to a city or airport characterized by several features:

- Longitude (`lon`)
- Latitude (`lat`)
- Population
- Country (categorical, one-hot encoded)
- City name (string attribute)

After preprocessing, we obtained **3363 nodes** and **27,094 undirected edges**. Each node is described by a **215-dimensional feature vector** combining numerical and categorical attributes. The task is **link prediction**: predicting whether a connection (route) exists between two airports using graph-based methods.

***

### 2. Models and Baselines

We compared **learning-based methods** (GAE and VGAE) against **classical topological heuristics** commonly used for link prediction.

#### 2.1 Learning-based models

- **Graph Autoencoder (GAE)**:  
  A deterministic encoder-decoder model using two GCN layers for node embedding and a dot-product decoder for edge reconstruction.

- **Variational Graph Autoencoder (VGAE)**:  
  Similar to GAE but introduces Gaussian latent variables (`μ`, `logσ`) and a Kullback-Leibler (KL) divergence term for probabilistic embedding regularization.

#### 2.2 Classical heuristics

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

### 3. Experimental Setup

- **Data split:** 80% train edges, 20% test edges (using `train_test_split_edges` from PyTorch Geometric)
- **Latent dimension:** 16
- **Optimizer:** Adam
- **Learning rates tested:** 0.01, 0.001
- **Epochs:** 600
- **Evaluation metrics:** Area Under ROC Curve (AUC) and Average Precision (AP)

Both AUC and AP were computed on the test link set (`test_pos_edge_index`, `test_neg_edge_index`).

***

### 4. Results

| Model | AUC | AP |
|--------|------|------|
| **Jaccard Coefficient** | **0.9329** | **0.9298** |
| **Preferential Attachment** | 0.9069 | 0.9165 |
| **Graph Autoencoder (GAE)** | 0.8471 | 0.8515 |
| **Variational Graph Autoencoder (VGAE)** | 0.8066 | 0.7889 |

***

### 5. Discussion

#### 5.1 Observations
1. **Topological heuristics significantly outperform neural models.**  
   The Jaccard coefficient achieves an AUC of 0.93 and AP of 0.92, clearly surpassing GAE (0.85) and VGAE (0.79). This shows that the connectivity structure of the airport network is highly predictable using local neighborhood overlap alone.

2. **GAE outperforms VGAE.**  
   The deterministic GAE performs better than the variational model. The KL divergence term in VGAE adds stochastic regularization, which helps on noisy data but degrades performance on this clean, deterministic network.

3. **Node features add limited information.**  
   Attributes like latitude, longitude, and population do not strongly determine whether two airports are connected. Hence, structural heuristics capture the essential link patterns better than learned embeddings.

***

### 6. Interpretation

The discrepancy between classical and learned models stems from **the nature of the data**:

- **Connectivity in this graph is topological**, not semantic. Airports connect mainly based on *existing route overlap* and *hub size*, patterns directly measured by Jaccard and PA.
- GNN-based models (GAE, VGAE) aggregate node features, but these features (geographic or demographic) do not drive link formation. Their embeddings, therefore, underperform on a purely structural task.
- Additionally, VGAE’s probabilistic latent space introduces unnecessary noise, while a simple local similarity metric like Jaccard remains sharp and accurate.

***

### 7. Future Work

To improve neural model performance, future experiments could:
- Include **edge-level features** such as geographic distance or passenger flow.
- Use deeper or alternative GNN architectures (e.g., GraphSAGE, GAT).
- Combine heuristic scores (like Jaccard) with node embeddings to form hybrid models.
- Perform ablation studies on features and GCN depth.

***

### 8. Conclusion

This study shows that for highly structural graphs like the airport network, **classical heuristics remain superior for link prediction**, achieving higher AUC and AP than GAE and VGAE.  
Neural methods are more valuable when **node attributes carry meaningful relational information** or when **latent non-linear dependencies** exist in the data.  

Nevertheless, the experiments validate the correctness of the GAE/VGAE pipeline and highlight the importance of matching model complexity with the underlying graph properties.




Using an identity matrix like `np.eye(df.shape[0], dtype=np.float32)` as node features in GNNs means exactly that each node is represented by a **unique one-hot vector** with no other information. Here is an extended explanation:

### What it means conceptually

- Each node gets a vector all zeros except for one position (its own index), which is 1.
- This is like saying "The only feature of this node is that it's node number i."
- Thus, the feature vectors don't carry any attribute information (like location or population) but uniquely identify nodes as distinct inputs.

### What happens in your GNN with identity features

- The graph convolution layers multiply these identity features by weight matrices and aggregate neighbor information using the adjacency matrix.
- Since the features are unique for each node and don't carry semantic info, the embeddings come mainly from the **graph structure through message passing**.
- The network learns embeddings purely from connectivity without help from node attributes.

### Why use identity features?

- To simulate **"no feature"** scenario so you can measure how much your model depends on node attributes versus pure graph structure.
- Helps to isolate the effect of graph topology in link prediction or node classification tasks.
- If performance with identity features is similar to or better than using real node features, it suggests the features add little useful signal.

### Analogies

- Identity matrix features are like "one-hot encoded node IDs".
- Without additional features, your model uses graph **connectivity patterns alone** to learn representations.


means giving your model **only unique node IDs and no attribute data**. This tests how well graph neural networks can predict links based purely on topology without attribute guidance.

This is a common baseline to evaluate the contribution and necessity of node features in graph ML tasks.

