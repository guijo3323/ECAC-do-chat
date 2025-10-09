import numpy as np
from sklearn.cluster import KMeans, DBSCAN

def magnitude(x, y, z):
    return np.sqrt(x*x + y*y + z*z)

def zscore(x):
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x)
    return (x - mu) / sd

def zscore_outliers(x, k=3.0):
    z = zscore(x)
    return np.abs(z) > k, z

def density_outliers(mask):
    n_out = np.count_nonzero(mask)
    n_tot = mask.size
    return (n_out / n_tot) * 100.0 if n_tot > 0 else 0.0

def kmeans_outlier_labels(X, n_clusters=5):
    X = np.asarray(X, dtype=float)
    if X.shape[0] < n_clusters:
        raise ValueError("Número de amostras inferior ao número de clusters")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_[labels]
    d = np.linalg.norm(X - centers, axis=1)
    thr = np.quantile(d, 0.95)
    mask = d > thr
    return mask, labels, d, thr

def dbscan_outlier_labels(X, eps=0.5, min_samples=20):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    mask = labels == -1
    return mask, labels
