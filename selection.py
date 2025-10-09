import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def fit_pca(X, variance=0.75):
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    pca = PCA(n_components=variance, svd_solver="full", random_state=42)
    Xp = pca.fit_transform(Xn)
    return scaler, pca, Xp

def fisher_score(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    classes = np.unique(y)
    n, d = X.shape
    mu = np.mean(X, axis=0)
    num = np.zeros(d)
    den = np.zeros(d) + 1e-12
    for c in classes:
        Xc = X[y==c]
        nc = Xc.shape[0]
        muc = np.mean(Xc, axis=0)
        varc = np.var(Xc, axis=0) + 1e-12
        num += nc * (muc - mu)**2
        den += nc * varc
    F = num / den
    return F

def relieff(X, y, n_neighbors=10, n_samples=None, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n, d = X.shape
    if n_samples is None or n_samples > n:
        n_samples = n
    if n <= 1:
        return np.zeros(d, float)
    n_neighbors = max(1, min(n_neighbors, n - 1))
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    scores = np.zeros(d, float)
    idxs = rng.choice(n, size=n_samples, replace=False)
    classes = np.unique(y)
    for i in idxs:
        xi, yi = X[i], y[i]
        dists, neighs = nn.kneighbors([xi], return_distance=True)
        neighs = neighs[0][1:]
        hits = [j for j in neighs if y[j] == yi][:n_neighbors]
        misses_by_class = {c: [j for j in neighs if y[j]==c][:n_neighbors] for c in classes if c!=yi}
        if hits:
            diffH = np.abs(X[hits] - xi).mean(axis=0)
            scores -= diffH / (n_samples)
        for c, js in misses_by_class.items():
            if js:
                Pc = np.mean(y==c)
                diffM = np.abs(X[js] - xi).mean(axis=0)
                scores += (Pc * diffM) / (n_samples)
    return scores
