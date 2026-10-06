# Lesson — L05 Unsupervised Learning

> **Sarah Chen's sixth day at NorthStar Retail.**
> Marcus signed off on the churn model from L04, then asked: *"Can we find natural CLUSTERS of customer behaviour — without labels?"* Sarah opens the same `northstar_customers.csv` she used in L03/L04, drops the `churned` column, and has to ship three things by the end of the day: a picture of the customer base, named customer segments, and a watch list of the unusual ones.

This document is a **short reference**. The lesson itself is taught in the notebooks. Read this for orientation before class, then come back to it for the takeaways, the technique-choice checklist, the review questions and the course map.

---

## How L05 is taught

| Stage | Where to go |
|---|---|
| **Pre-class** | [`pre-class.md`](./pre-class.md) + `notebooks/01_morning_briefing.ipynb` |
| **In-class — Part 1: PCA** | `notebooks/02_pca.ipynb` |
| **In-class — Part 2: K-Means** | `notebooks/03_kmeans.ipynb` |
| **In-class — Part 3: Isolation Forest** | `notebooks/04_isolation_forest.ipynb` |
| **Self-study** | `notebooks/assignment.ipynb` + `notebooks/optional_extensions.ipynb` |
| **Glossary & further reading** | [`reference.md`](./reference.md) |
| **Takeaways & review** | This document |

The notebooks are the spine. Run them in order, then come back here to consolidate.

---

## Overview

In L03 and L04 Sarah always had a target column. Today there isn't one, and Marcus's brief is really two jobs: *find groups* and *find oddballs*. The three Parts of the lesson answer it in order:

| Part | Technique | Job | What Sarah gets from NorthStar's 10,000 customers |
|---|---|---|---|
| 1 | **PCA** | Compress many features into a few axes | 10 features become 17 columns after preprocessing. The first 2 PCs show only **22%** of the variance, and it takes **8 PCs to reach 80%**. |
| 2 | **K-Means** | Group similar rows into segments | **K = 4** segments: Loyal Veterans (35%), Dormant / At-Risk (11%), New Actives (35%), Serial Returners (19%) |
| 3 | **Isolation Forest** | Rank rows by how unusual they are | A **watch list of 500** customers (the 5% most isolated) |

Same dataset and same preprocessing as L03/L04: median-impute, standard-scale, one-hot encode. What changes is that there is no "right answer" to score against. PCA's 22% is not a failure, and the elbow plot is not broken. Both are honest descriptions of data that has no sharp structure, and your job is to say so while still delivering something the business can use.

---

## Key takeaways

1. **Unsupervised learning is three jobs, not one.** Dimensionality reduction (PCA), clustering (K-Means) and anomaly detection (Isolation Forest) cover most real unsupervised work. Pick the job before you pick the algorithm.
2. **Scale before any distance-based method.** PCA maximises variance and K-Means minimises squared distance, so both are scale-sensitive. On unscaled NorthStar data, `avg_monthly_spend_gbp` (range £5–£500) takes PC1 almost entirely (loading 1.000). That is not "most informative", just biggest.
3. **PCA reads variance, not meaning.** `explained_variance_ratio_` says how much each component carries, and `components_` says which features drive it. Naming a PC is a human step done by reading the loadings. Also, the best axes for *drawing* clusters are not always the biggest ones. In NB 03, PC6 separates the four segments best (F ≈ 2,133, against 824 for PC1).
4. **A low variance-explained figure is a finding, not an error.** NorthStar's features are nearly uncorrelated (every pair has |r| ≤ 0.02), so variance is spread almost evenly across the first 8 PCs (~11% each). A 2D PCA plot is a *picture* of the data, not a summary of it. Clustering downstream still uses all 17 dimensions.
5. **K-Means will not tell you K, and the elbow plot often won't either.** On NorthStar, inertia falls smoothly from K = 2 to 10 and silhouette stays between 0.080 and 0.090. The defensible answer combines the metrics with what the business can act on. That is why the notebook picks K = 4 over the silhouette-best K = 6.
6. **K-Means assumes compact, similarly-sized clusters, and its answer depends on the seed.** Refitting with `random_state=123` instead of 42 gives a different partition (Adjusted Rand Index 0.569, where 1.0 would be identical). Treat segment boundaries as soft. A tiny cluster next to big ones is usually an anomaly group, not a segment.
7. **Cluster labels are arbitrary until you profile them.** `fit_predict` returns integers. The deliverable is *named segments*: group by cluster, compare each cluster's means to the global mean (Z-scores make this readable), and write one sentence per cluster.
8. **Isolation Forest's `contamination` only sets the threshold.** The continuous `score_samples()` output does not depend on it, so you can re-threshold after fitting without retraining. Treat `contamination` as an operational dial (1% flags 100 customers, 5% flags 500, 10% flags 1,000), not a model parameter.
9. **An anomaly is a starting point, not a verdict.** The model surfaces unusual feature *combinations*, and a human reads the row, decides what it means and routes it to the right team. In NB 04, 301 of the 500 flagged customers were caught by Isolation Forest but not by a simple |z| > 3 check, and 470 |z|-flagged customers were not in the 500.

---

## Choosing an unsupervised technique — a checklist

Before you reach for PCA, K-Means or Isolation Forest, run the question through this three-step check:

1. **What is the deliverable: a picture, a label per row, or a watch list?** A 2D picture for a stakeholder is PCA (or t-SNE if PCA looks featureless). A segment per customer is K-Means (or DBSCAN / hierarchical clustering if the shapes look non-globular). A ranked list of "weird" rows is Isolation Forest. The deliverable picks the algorithm, not the other way round.
2. **Is the data ready for a distance-based method?** Use the same preprocessing as a supervised model: median-impute, standard-scale, one-hot encode. In today's notebooks the pipeline is fitted on all 10,000 customers, because the goal is to describe *this* customer base and nothing is held out. If you will later score *new* customers, fit the imputer and scaler on your reference data only and `transform` the newcomers. Fitting on everything would let future data shape the scale. There is no test score to warn you if you get this wrong, so unsupervised work needs the discipline *more*, not less.
3. **Can you defend the choice with both a metric and a sentence?** "K = 4 because silhouette peaks there" is wrong on NorthStar, because silhouette peaks at K = 6. The answer Marcus accepts is "K = 4 because its silhouette (0.088) is within 0.002 of the best, and marketing can run four campaigns but not six."

Skip any of these and you will ship a result that is mathematically defensible but operationally useless, or one the stakeholder cannot read.

---

## Check your understanding

Work through these after finishing the three Part notebooks. Attempt each question on your own first. Numbers in the sample answers come from the NorthStar notebooks. Questions marked *hypothetical* describe a situation you may meet on other data.

### Part 1 — PCA

**Q1 — Why scale first?** A colleague runs PCA directly on the raw NorthStar features and finds PC1 is almost entirely `avg_monthly_spend_gbp`. What happened, and what should they do?

> **Sample answer:** PCA maximises variance, and spend (£5–£500) has a far bigger numeric range than `returns_per_purchase` (0–0.5) or `support_tickets_quarter` (0–7). Unscaled, its raw variance swamps everything else. In NB 02 the PC1 loading on spend is 1.000 and every other feature is ≤ 0.005. Fix: `StandardScaler` before `PCA`, so each feature contributes on equal footing and PC1 reflects the strongest *joint* pattern rather than the loudest column.

**Q2 — Reading the scree plot.** On NorthStar the first two PCs explain 22% of total variance, and you need 8 components for 80% (10 for 90%). What does this tell you about the data, and what do you say when Marcus sees the 2D plot?

> **Sample answer:** The features are mostly independent, so variance is spread across many directions (the first 8 PCs each carry about 11%) and PCA cannot compress aggressively. The 2D plot is still a useful picture, but be honest that it shows only ~22% of the signal. The cost is real: in NB 02, a churn model on 2 PCs scores CV F1 0.204, against 0.328 for all 17 columns. Clustering downstream uses all 17 dimensions, not the two on the screen.

**Q3 — Naming a component.** PC1's biggest loadings are `support_tickets_quarter` (+0.56), `age` (+0.45), `avg_review_polarity` (−0.43) and `tenure_months` (+0.35). What is PC1 about, and how would you describe it to a non-technical stakeholder?

> **Sample answer:** PC1 separates older, longer-tenured customers who raise many support tickets and leave lower reviews from newer, younger customers who raise few and review more warmly. It is a "long-standing but frustrated" versus "newer and content" axis. Spend (+0.11) barely contributes. The algorithm gives coefficients and you supply the label, and with only ~11% of variance on PC1 you should present the label as a tendency, not a hard rule.

### Part 2 — K-Means

**Q4 — When the elbow is smooth.** On NorthStar, inertia falls smoothly from K = 2 to 10 with no clear bend. Silhouette peaks at K = 6 (0.0902), while K = 4 scores 0.0880. What do you ship?

> **Sample answer:** Ship K = 4, and say why. A 0.002 silhouette gap is noise at this dataset size, the curves show no strong natural clustering, and four segments are a number marketing can brief and name (six are not). The honest framing is "chosen for actionability, not statistical separation". If neither the metrics nor the business point to a K, try another algorithm (DBSCAN for density, hierarchical for nested structure). Do not pretend the elbow showed something it did not.

**Q5 — A tiny cluster (hypothetical).** On some other dataset, K-Means with K = 4 returns three clusters of ~2,500 customers and one with 12. What is the most likely interpretation? (NorthStar's real sizes are 3,463 / 1,138 / 3,540 / 1,859, so you will not see this there.)

> **Sample answer:** The 12-customer cluster is probably an anomaly group, a few customers so different from everyone else that K-Means gave them their own centroid. Either report them as a watch list alongside the three real segments, or set them aside and refit with K = 3. Do not present "four equally important segments", because that misrepresents the result.

**Q6 — Naming the clusters.** After fitting K = 4, how do you turn the integer labels into something Marcus can act on?

> **Sample answer:** Group by cluster, compute each feature's mean, and compare it to the global mean (or use Z-scores for a heatmap). Then write one sentence per cluster from its profile. On NorthStar the dominant signals are tenure, login recency and return rate, not spend or age:
>
> | Cluster | Size | Profile | Name |
> |---|---|---|---|
> | 0 | 3,463 (35%) | tenure +0.91 SD, returns −0.43 SD | Loyal Veterans |
> | 1 | 1,138 (11%) | last login ~92 days vs 30 overall (+2.07 SD) | Dormant / At-Risk |
> | 2 | 3,540 (35%) | tenure −0.91 SD, returns −0.41 SD | New Actives |
> | 3 | 1,859 (19%) | returns 0.31 vs 0.14 overall (+1.60 SD) | Serial Returners |
>
> The deliverable is the names plus the profile table, not the integers.

### Part 3 — Isolation Forest

**Q7 — What `contamination` controls.** You fit Isolation Forest with `contamination=0.05` and then decide 5% is too aggressive, because you only want about 2%. Do you have to refit?

> **Sample answer:** No. `contamination` only sets the cutoff between the +1 and −1 labels, and `score_samples()` does not depend on it. Keep the fitted model and pick a stricter cutoff, such as the 2nd-percentile score. In NB 04, taking the 100 lowest scores (cutoff −0.571) gives exactly the same 100 customers as `contamination=0.01`. This is why `contamination` is an operational dial: set it to the number of customers your team has capacity to review.

**Q8 — Why a flagged row was flagged.** The most anomalous NorthStar customer (score −0.62) is 67 years old with 18 months' tenure, spends £438 a month and last logged in 170 days ago. What makes this row unusual, and what do you do with it?

> **Sample answer:** Compare the row to the global medians: £438 is about 8× the median spend (£53.90), and 170 days is about 8× the median time since login (21 days). Neither is extreme alone. The *combination* (a big spender who has gone quiet) is what makes it easy to isolate. The action is not "label as fraud". Route it to customer success or marketing for a win-back call, and send the diverging features with the flag so the recipient knows where to start. The model surfaces the row and a human investigates.

**Q9 — Anomalies vs segments.** After fitting both K-Means (K = 4) and Isolation Forest, you count anomalies per cluster and get 105, 158, 92 and 145. What does this show?

> **Sample answer:** Anomalies appear in every cluster, so Isolation Forest is not just picking out the fringe of one segment. Clustering and anomaly detection are answering different questions: clusters are dense regions, and anomalies are sparse points that can sit at the edge of any of them. But the *rates* are uneven: 3.0% (Loyal Veterans), 13.9% (Dormant / At-Risk), 2.6% (New Actives) and 7.8% (Serial Returners), against 5% overall. Dormant customers are flagged almost three times as often as average, so start the watch list there. The two methods are complementary: segments for marketing, a ranked watch list for customer success.

---

## Where L05 fits in the course

L05 is the first lesson without labels. The same techniques come back whenever the target column is missing, noisy or arrives late.

| Lesson | How L05 shows up |
|---|---|
| **L06 — Time Series** | Marcus's next question. Forecasting (ETS, ARIMA, ML-based) comes first, and anomaly detection on forecast residuals reuses the Isolation Forest idea. |
| **L07 — Neural Networks** | Autoencoders generalise PCA to non-linear compression, and reconstruction error works as an anomaly score. |
| **L08 — Computer Vision** | Image embeddings have hundreds of dimensions, and PCA (or t-SNE / UMAP) is the standard way to look at them. Defect detection is anomaly detection on image features. |
| **L09 — NLP & Embeddings** | Document embeddings can be clustered with K-Means to find topics without labelled categories. Near-duplicate detection is anomaly detection in embedding space. |
| **L10 — Transformers & GenAI** | Semantic search is a distance computation in a learned space, so the metric intuitions from K-Means apply. Checking generations against a reference set can use anomaly scoring. |

---

> *Marcus nods.* **"Good. Now — sales are seasonal. Can you forecast next quarter's revenue?"**
>
> That question (*can I predict what comes next when the data has a time order?*) is the engine of **L06 (Time Series Forecasting)**.
