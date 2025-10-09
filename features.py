import numpy as np
from scipy.stats import skew, kurtosis, iqr
from scipy.signal import welch

def time_features(sig):
    x = np.asarray(sig)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "var": np.nan, "rms": np.nan,
                "iqr": np.nan, "mad": np.nan, "skew": np.nan, "kurt": np.nan, "zcr": np.nan}
    mean = float(np.mean(x))
    std  = float(np.std(x))
    var  = float(np.var(x))
    rms  = float(np.sqrt(np.mean(x*x)))
    iqrv = float(iqr(x)) if x.size>1 else 0.0
    mad  = float(np.mean(np.abs(x - mean)))
    sk   = float(skew(x)) if x.size>2 else 0.0
    ku   = float(kurtosis(x)) if x.size>3 else 0.0
    zcr  = float(np.mean((x[:-1]*x[1:] < 0))) if x.size>1 else 0.0
    return {"mean":mean,"std":std,"var":var,"rms":rms,"iqr":iqrv,"mad":mad,"skew":sk,"kurt":ku,"zcr":zcr}

def spectral_features(sig, fs=50.0):
    x = np.asarray(sig)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return {"spec_centroid": np.nan, "spec_bw": np.nan, "spec_entropy": np.nan, "power": np.nan}
    f, Pxx = welch(x, fs=fs, nperseg=min(256, x.size))
    Pxx = np.maximum(Pxx, 1e-12)
    power = float(np.sum(Pxx))
    sc = float(np.sum(f * Pxx) / np.sum(Pxx))
    bw = float(np.sqrt(np.sum(((f - sc)**2) * Pxx) / np.sum(Pxx)))
    p = Pxx / np.sum(Pxx)
    ent = float(-np.sum(p * np.log2(p)))
    ent_norm = float(ent / np.log2(p.size))
    return {"spec_centroid": sc, "spec_bw": bw, "spec_entropy": ent_norm, "power": power}

def windowed_features(arr, cols_xyz, fs=50.0, window=256, hop=128):
    X, Y = [], []
    x = arr[:, cols_xyz[0]]
    y = arr[:, cols_xyz[1]]
    z = arr[:, cols_xyz[2]]
    lab = arr[:, 11]
    mag = np.sqrt(x*x + y*y + z*z)
    N = arr.shape[0]
    for start in range(0, N - window + 1, hop):
        end = start + window
        segs = {"x": x[start:end], "y": y[start:end], "z": z[start:end], "mag": mag[start:end]}
        feat_vec = []
        for key in ["x","y","z","mag"]:
            tf = time_features(segs[key])
            sf = spectral_features(segs[key], fs=fs)
            for k in ["mean","std","var","rms","iqr","mad","skew","kurt","zcr"]:
                feat_vec.append(tf[k])
            for k in ["spec_centroid","spec_bw","spec_entropy","power"]:
                feat_vec.append(sf[k])
        ywin = lab[start:end]
        if ywin.size == 0:
            continue
        vals, counts = np.unique(ywin, return_counts=True)
        ylab = vals[np.argmax(counts)]
        X.append(feat_vec)
        Y.append(ylab)
    X_arr = np.asarray(X, dtype=float)
    if X_arr.size == 0:
        feat_len = 4 * (9 + 4)
        X_arr = np.empty((0, feat_len), dtype=float)
    Y_arr = np.asarray(Y, dtype=int)
    return X_arr, Y_arr
