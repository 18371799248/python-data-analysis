import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据集并标准化
data = load_iris()
X = data.data
y_true = data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 手肘法计算不同K的WCSS
wcss = []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_range, wcss, marker='o')
plt.xlabel("K值")
plt.ylabel("WCSS")
plt.title("手肘法曲线")
plt.grid(True)
plt.savefig("elbow.png", dpi=150)
plt.show()

# K=3 K‑Means训练，计算轮廓系数
best_k = 3
km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
labels_km = km.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, labels_km)
sil_vals = silhouette_samples(X_scaled, labels_km)
print(f"K‑Means K=3 轮廓系数：{sil_score:.4f}")

# 绘制轮廓刀片图
plt.figure(figsize=(7,5))
y_lower = 10
for i in range(best_k):
    cluster_sil = sil_vals[labels_km == i]
    cluster_sil.sort()
    size = len(cluster_sil)
    y_upper = y_lower + size
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil)
    y_lower = y_upper + 10
plt.axvline(x=sil_score, color="red", linestyle="--")
plt.title("轮廓系数刀片图")
plt.xlabel("轮廓系数")
plt.ylabel("簇")
plt.savefig("silhouette.png", dpi=150)
plt.show()

# PCA降维到二维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1,2,figsize=(11,5))
axes[0].scatter(X_pca[:,0], X_pca[:,1], c=labels_km, cmap="Set1")
axes[0].set_title("K‑Means (PCA降维后)")

db = DBSCAN(eps=0.6, min_samples=5)
labels_db = db.fit_predict(X_scaled)
axes[1].scatter(X_pca[:,0], X_pca[:,1], c=labels_db, cmap="Set1")
axes[1].set_title("DBSCAN (PCA降维后)")
plt.savefig("cluster_pca.png", dpi=150)
plt.show()

print("DBSCAN 标签中 -1 代表噪声样本")
n_cluster = len(set(labels_db)) - (1 if -1 in labels_db else 0)
print(f"DBSCAN识别得到簇数量：{n_cluster}")
