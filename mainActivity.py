import argparse
import numpy as np
from data_io import load_all, COLUMNS
from outliers import magnitude, zscore_outliers, density_outliers, kmeans_outlier_labels, dbscan_outlier_labels
from plotting import boxplot_by_activity, scatter_mark_outliers, plot_3d
from features import windowed_features
from selection import fit_pca, fisher_score, relieff
from sklearn.preprocessing import StandardScaler

def _select_signal(arr, signal):
    if signal == "accel":
        cols = (COLUMNS["accel_x"], COLUMNS["accel_y"], COLUMNS["accel_z"])
    elif signal == "gyro":
        cols = (COLUMNS["gyro_x"], COLUMNS["gyro_y"], COLUMNS["gyro_z"])
    elif signal == "magnet":
        cols = (COLUMNS["mag_x"], COLUMNS["mag_y"], COLUMNS["mag_z"])
    else:
        raise ValueError("signal deve ser accel|gyro|magnet")
    return cols

def cmd_boxplots(args):
    arr = load_all(args.data_root)
    cols = _select_signal(arr, args.signal)
    m = magnitude(arr[:,cols[0]], arr[:,cols[1]], arr[:,cols[2]])
    labels = arr[:, 11].astype(int)
    boxplot_by_activity(m, labels, title=f"Boxplots módulos ({args.signal})", savefig=args.savefig)

def cmd_outlier_density(args):
    arr = load_all(args.data_root)
    arr = arr[arr[:,0] == 2]  # device 2 (pulso direito)
    cols = _select_signal(arr, args.signal)
    m = magnitude(arr[:,cols[0]], arr[:,cols[1]], arr[:,cols[2]])
    mask, _ = zscore_outliers(m, k=3.0)
    dens = density_outliers(mask)
    print(f"Densidade de outliers (%): {dens:.3f}")

def cmd_mark_z(args):
    arr = load_all(args.data_root)
    cols = _select_signal(arr, args.signal)
    m = magnitude(arr[:,cols[0]], arr[:,cols[1]], arr[:,cols[2]])
    mask, _ = zscore_outliers(m, k=args.k)
    scatter_mark_outliers(m, mask, title=f"Z-score k={args.k} — {args.signal}", savefig=args.savefig)

def cmd_kmeans_outliers(args):
    arr = load_all(args.data_root)
    cols = _select_signal(arr, args.signal)
    X3 = arr[:, [cols[0], cols[1], cols[2]]]
    mask, labels, d, thr = kmeans_outlier_labels(X3, n_clusters=args.n_clusters)
    print(f"threshold dist p95: {thr:.4f}  | outliers: {mask.sum()} / {mask.size} ({100*mask.mean():.2f}%)")
    plot_3d(X3, mask=mask, title=f"k-means outliers ({args.signal})", savefig=args.savefig)

def cmd_dbscan_outliers(args):
    arr = load_all(args.data_root)
    cols = _select_signal(arr, args.signal)
    X3 = arr[:, [cols[0], cols[1], cols[2]]]
    mask, labels = dbscan_outlier_labels(X3, eps=args.eps, min_samples=args.min_samples)
    print(f"ruído (outliers DBSCAN): {mask.sum()} / {mask.size} ({100*mask.mean():.2f}%)")
    plot_3d(X3, mask=mask, title=f"DBSCAN outliers ({args.signal})", savefig=args.savefig)

def cmd_extract_features(args):
    arr = load_all(args.data_root)
    cols = _select_signal(arr, "accel")
    X, y = windowed_features(arr, cols, fs=args.fs, window=args.window, hop=args.hop)
    np.savez(args.save, X=X, y=y)
    print(f"Guardado {args.save}  | X: {X.shape}  y: {y.shape}")

def cmd_pca(args):
    data = np.load(args.features)
    X, y = data["X"], data["y"]
    scaler, pca, Xp = fit_pca(X, variance=args.variance)
    exp = pca.explained_variance_ratio_.cumsum()
    k = np.searchsorted(exp, args.variance) + 1
    print(f"Componentes necessárias para {args.variance:.0%} de variância explicada: {k}")
    x0 = X[0:1]
    x0n = scaler.transform(x0)
    x0p = pca.transform(x0n)
    print("Exemplo (instante idx=0) features comprimidas:", x0p[0,:k])

def cmd_fisher(args):
    data = np.load(args.features)
    X, y = data["X"], data["y"]
    Xn = StandardScaler().fit_transform(X)
    F = fisher_score(Xn, y)
    idx = np.argsort(F)[::-1][:args.top_k]
    print("Top features (Fisher):", idx.tolist())
    print("Scores:", np.round(F[idx], 4).tolist())

def cmd_relief(args):
    data = np.load(args.features)
    X, y = data["X"], data["y"]
    Xn = StandardScaler().fit_transform(X)
    R = relieff(Xn, y, n_neighbors=args.n_neighbors, n_samples=args.n_samples or None)
    idx = np.argsort(R)[::-1][:args.top_k]
    print("Top features (ReliefF):", idx.tolist())
    print("Scores:", np.round(R[idx], 4).tolist())

def build_parser():
    p = argparse.ArgumentParser(description="EA/ECAC 2025 — HAR pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("boxplots")
    pb.add_argument("--data_root", required=True)
    pb.add_argument("--signal", choices=["accel","gyro","magnet"], required=True)
    pb.add_argument("--savefig", default=None)
    pb.set_defaults(func=cmd_boxplots)

    pdens = sub.add_parser("outlier-density")
    pdens.add_argument("--data_root", required=True)
    pdens.add_argument("--signal", choices=["accel","gyro","magnet"], required=True)
    pdens.set_defaults(func=cmd_outlier_density)

    pmz = sub.add_parser("mark-z")
    pmz.add_argument("--data_root", required=True)
    pmz.add_argument("--signal", choices=["accel","gyro","magnet"], required=True)
    pmz.add_argument("--k", type=float, default=3.0)
    pmz.add_argument("--savefig", default=None)
    pmz.set_defaults(func=cmd_mark_z)

    pk = sub.add_parser("kmeans-outliers")
    pk.add_argument("--data_root", required=True)
    pk.add_argument("--signal", choices=["accel","gyro","magnet"], required=True)
    pk.add_argument("--n_clusters", type=int, default=5)
    pk.add_argument("--savefig", default=None)
    pk.set_defaults(func=cmd_kmeans_outliers)

    pdb = sub.add_parser("dbscan-outliers")
    pdb.add_argument("--data_root", required=True)
    pdb.add_argument("--signal", choices=["accel","gyro","magnet"], required=True)
    pdb.add_argument("--eps", type=float, default=0.5)
    pdb.add_argument("--min_samples", type=int, default=20)
    pdb.add_argument("--savefig", default=None)
    pdb.set_defaults(func=cmd_dbscan_outliers)

    pf = sub.add_parser("extract-features")
    pf.add_argument("--data_root", required=True)
    pf.add_argument("--fs", type=float, default=50.0)
    pf.add_argument("--window", type=int, default=256)
    pf.add_argument("--hop", type=int, default=128)
    pf.add_argument("--save", default="features.npz")
    pf.set_defaults(func=cmd_extract_features)

    pp = sub.add_parser("pca")
    pp.add_argument("--features", required=True)
    pp.add_argument("--variance", type=float, default=0.75)
    pp.set_defaults(func=cmd_pca)

    pfi = sub.add_parser("fisher")
    pfi.add_argument("--features", required=True)
    pfi.add_argument("--top_k", type=int, default=10)
    pfi.set_defaults(func=cmd_fisher)

    pr = sub.add_parser("relieff")
    pr.add_argument("--features", required=True)
    pr.add_argument("--top_k", type=int, default=10)
    pr.add_argument("--n_neighbors", type=int, default=10)
    pr.add_argument("--n_samples", type=int, default=None)
    pr.set_defaults(func=cmd_relief)

    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
