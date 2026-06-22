"""
Predict movement-type labels for original empty world CAR trajectories.

Outputs in --out_dir:
  - full_features_raw.csv
  - full_features_norm.csv
  - full_predictions.csv   (per-class probs, plus pred & pred_conf)
  - full_pred_counts.csv
  - errors.csv             (files that failed to parse or were too short)
  - run.json               (manifest)

No aliasing: 0001000 is not used or mapped. Windows-friendly; defaults to --workers 1.
"""

import os, re, math, json, time, argparse, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 40-feature whitelist (order matters; we’ll respect features.json order when scaling)
KEEP_FEATURES = [
    "ac1_step","ac2_turns","axial_R2","axis_adherence_15deg","cv_step","dir_entropy","dom_freq",
    "flatness","hull_area_norm","hull_circularity","kurt_turn","lin_corner_changes","linear_r2",
    "low_high_ratio","mad_step","max_abs_turn","mean_turn","occupancy_entropy","orientation_eff",
    "oscillation_rate","pc_var_ratio","peak_count_per100","peak_prom_cv","peak_prom_median",
    "peak_regularity","peak_spacing_cv","peak_spacing_mean","polar_hull_ratio","poly_rmse_deg1",
    "quad_R4","rdp_retention_1pct","rdp_retention_2pct","recurrence_ratio","run_len_mean","skew_turn",
    "spec_entropy","std_turn","straightness","time_to_half_coverage","turn_consistency"
]

# -------- IO helpers --------
def read_xy(csv_path: str):
    """Read Final X/Y columns from a trajectory CSV (robust to name variants)."""
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}

    # prefer explicit final_x/final_y if present
    cx = cols.get("final_x") or next((c for k, c in cols.items() if "final_x" in k), None)
    cy = cols.get("final_y") or next((c for k, c in cols.items() if "final_y" in k), None)

    # fallbacks: last *_x / *_y column
    if cx is None:
        x_candidates = [c for k, c in cols.items() if k.endswith("_x")]
        if x_candidates: cx = x_candidates[-1]
    if cy is None:
        y_candidates = [c for k, c in cols.items() if k.endswith("_y")]
        if y_candidates: cy = y_candidates[-1]

    if cx is None or cy is None:
        return None, None

    x = pd.to_numeric(df[cx], errors="coerce").dropna().values
    y = pd.to_numeric(df[cy], errors="coerce").dropna().values
    if len(x) < 3 or len(x) != len(y):
        return None, None
    return x, y

# -------- math / metrics (same as training) --------
def _wrap_pi(x): return (x + np.pi) % (2*np.pi) - np.pi

def _headings(x, y):
    dx = np.diff(x); dy = np.diff(y)
    if dx.size == 0: return np.array([])
    return np.arctan2(dy, dx)

def _turns(x, y):
    a = _headings(x,y)
    if a.size < 2: return np.array([])
    return _wrap_pi(a[1:] - a[:-1])

def _path_len(x,y): return float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if len(x)>1 else 0.0
def _net_disp(x,y): return float(np.hypot(x[-1]-x[0], y[-1]-y[0])) if len(x)>1 else 0.0

def _rotate_to_main(x, y):
    dx = x[-1]-x[0]; dy = y[-1]-y[0]
    ang = math.atan2(dy, dx)
    rx = np.array(x) - x[0]; ry = np.array(y) - y[0]
    cx =  np.cos(-ang)*rx - np.sin(-ang)*ry
    cy =  np.sin(-ang)*rx + np.cos(-ang)*ry
    return cx, cy, ang

def _detrended_perp(x, y):
    rx, ry, _ = _rotate_to_main(x,y)
    n = len(ry); trend = np.linspace(ry[0], ry[-1], n)
    return ry - trend

def _convex_hull(points):
    pts = np.unique(points, axis=0)
    if len(pts) < 3: return pts, 0.0, 0.0
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]; upper=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p) <= 0: lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper)>=2 and cross(upper[-2],upper[-1],p) <= 0: upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1]+upper[:-1])
    x=hull[:,0]; y=hull[:,1]
    area = 0.5*np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
    per  = np.sum(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0])))
    return hull, float(area), float(per)

def _linear_r2(x, y):
    if len(x) < 3: return 0.0
    X = np.vstack([x, np.ones_like(x)]).T
    a_b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ a_b
    ss_res = float(((y - yhat)**2).sum())
    ss_tot = float(((y - y.mean())**2).sum()) + 1e-12
    return max(0.0, 1.0 - ss_res/ss_tot)

def _axis_adherence(theta, deg=15):
    mod = np.mod(np.abs(theta), np.pi/2)
    d2axis = np.minimum(mod, (np.pi/2)-mod)
    return float(np.mean(d2axis < (deg*np.pi/180)))

def _fft_features(signal):
    s = np.asarray(signal, dtype=float)
    n = len(s)
    if n < 8:
        return dict(dom_freq=0.0, spec_entropy=0.0, low_high_ratio=0.0, flatness=0.0)
    s = s - s.mean()
    N = 1 << (n-1).bit_length()
    S = np.fft.rfft(s, n=N)
    P = (S*np.conj(S)).real
    freqs = np.fft.rfftfreq(N, d=1.0)
    P[0]=0.0
    total = P.sum() + 1e-12
    if total == 0:
        return dict(dom_freq=0.0, spec_entropy=0.0, low_high_ratio=0.0, flatness=0.0)
    kmax = int(np.argmax(P)); dom_f = float(freqs[kmax])
    p = P/total
    spec_ent = float(-(p[p>0]*np.log(p[p>0])).sum()/np.log(len(p)))
    split = int(len(P)/3)
    lhr = float(P[:split].sum()/(P[split:].sum()+1e-12))
    gm = math.exp(np.log(P[P>0]).mean()) if np.any(P>0) else 0.0
    am = P.mean()
    flat = float(gm/(am+1e-12))
    return dict(dom_freq=dom_f, spec_entropy=spec_ent, low_high_ratio=lhr, flatness=flat)

def _polyfit_r2_rmse(x, y, deg):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < max(6, deg+2): return 0.0, np.inf
    try:
        coef = np.polyfit(x, y, deg=deg)
        yhat = np.polyval(coef, x)
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - y.mean())**2)) + 1e-12
        r2 = max(0.0, 1.0 - ss_res/ss_tot)
        rmse = math.sqrt(ss_res / max(n - (deg+1), 1))
        return r2, rmse
    except Exception:
        return 0.0, np.inf

def _local_peaks(signal, window=5, prominence_frac=0.1):
    n = len(signal)
    if n < 3:
        return [], [], []
    s = np.asarray(signal)
    peaks, troughs = [], []
    for i in range(1, n-1):
        if s[i] > s[i-1] and s[i] > s[i+1]:
            peaks.append(i)
        if s[i] < s[i-1] and s[i] < s[i+1]:
            troughs.append(i)
    prom = np.zeros(n, dtype=float)
    std = np.std(s) if np.std(s) > 1e-12 else 1.0
    for idx in peaks:
        left = max(0, idx-window); right = min(n-1, idx+window)
        base = max(np.min(s[left:idx]) if idx-left>0 else s[idx],
                   np.min(s[idx+1:right+1]) if right-idx>0 else s[idx])
        prom[idx] = max(0.0, s[idx]-base)
    for idx in troughs:
        left = max(0, idx-window); right = min(n-1, idx+window)
        base = min(np.max(s[left:idx]) if idx-left>0 else s[idx],
                   np.max(s[idx+1:right+1]) if right-idx>0 else s[idx])
        prom[idx] = max(0.0, base - s[idx])
    thr = prominence_frac * std
    keep_peaks   = [i for i in peaks   if prom[i] >= thr]
    keep_troughs = [i for i in troughs if prom[i] >= thr]
    prom_list = [prom[i] for i in keep_peaks + keep_troughs]
    return keep_peaks, keep_troughs, prom_list

# -------- feature computation (returns ONLY the keep-list) --------
def compute_features_for_series(x, y):
    feats = {}
    n = len(x)
    if n < 3: return feats

    L = _path_len(x,y); D = _net_disp(x,y)
    a = _headings(x,y); t = _turns(x,y); at = np.abs(t)

    feats["straightness"] = float(D/(L+1e-12))
    feats["mean_turn"] = float(at.mean()) if at.size else 0.0
    feats["std_turn"]  = float(at.std(ddof=1)) if at.size>1 else 0.0
    feats["skew_turn"] = float(((at - at.mean())**3).mean() / (at.std()+1e-12)**3) if at.size>2 and at.std()>1e-12 else 0.0
    feats["kurt_turn"] = float(((at - at.mean())**4).mean() / (at.var()+1e-12)**2) if at.size>3 and at.var()>1e-12 else 0.0

    if a.size:
        C = np.mean(np.cos(a)); S = np.mean(np.sin(a))
        mean_dir = math.atan2(S, C)
        feats["orientation_eff"] = float(D / (np.sum(np.abs(_wrap_pi(a-mean_dir)))+1e-9))
    else:
        feats["orientation_eff"] = 0.0

    if t.size:
        signs = np.sign(t)
        p_left  = float(np.mean(signs>0)); p_right = float(np.mean(signs<0))
        feats["turn_consistency"] = float(max(p_left, p_right))
        runs=[]; run=1
        for i in range(1, len(signs)):
            if signs[i]==0 or signs[i]!=signs[i-1]: runs.append(run); run=1
            else: run+=1
        runs.append(run)
        feats["run_len_mean"] = float(np.mean(runs))
        feats["max_abs_turn"] = float(np.max(np.abs(t)))
    else:
        feats.update(dict(turn_consistency=0.0, run_len_mean=0.0, max_abs_turn=0.0))

    dx = np.diff(x); dy = np.diff(y); s = np.hypot(dx,dy)
    feats["cv_step"]  = float(s.std(ddof=1)/(s.mean()+1e-12)) if s.size>1 else 0.0
    feats["mad_step"] = float(np.median(np.abs(s-np.median(s)))) if s.size else 0.0
    feats["ac1_step"] = float(np.corrcoef(s[:-1], s[1:])[0,1]) if s.size>2 and np.std(s[:-1])>1e-12 and np.std(s[1:])>1e-12 else 0.0

    P = np.vstack([x,y]).T; Pc = P - P.mean(axis=0, keepdims=True)
    Cxy = np.cov(Pc.T); w,_ = np.linalg.eigh(Cxy); w = np.sort(np.maximum(w,0.0))
    feats["pc_var_ratio"] = float(w[-1]/w.sum()) if w.sum()>0 else 0.0
    _, area, per = _convex_hull(P)
    feats["hull_area_norm"]  = float(area/(D*D+1e-12))
    feats["hull_circularity"] = float((4*np.pi*area)/(per*per+1e-12)) if per>0 else 0.0
    feats["linear_r2"]       = _linear_r2(x,y)

    feats["axis_adherence_15deg"] = _axis_adherence(a, 15) if a.size else 0.0
    if a.size:
        close = 15*np.pi/180
        assign = np.full_like(a, 2, dtype=int)
        assign[np.abs(_wrap_pi(a))<close] = 0
        assign[np.abs(_wrap_pi(a - np.pi/2))<close] = 1
        comp=[assign[0]]
        for v in assign[1:]:
            if v!=comp[-1]: comp.append(v)
        feats["lin_corner_changes"] = float(sum((v in (0,1)) for v in comp))
    else:
        feats["lin_corner_changes"] = 0.0

    yp = _detrended_perp(x,y)
    if yp.size:
        pk, tr, prom = _local_peaks(yp, window=5, prominence_frac=0.1)
        idx = sorted(pk + tr)
        feats["peak_count_per100"] = float(100.0*len(idx)/max(len(yp),1))
        if len(idx)>=2:
            spac = np.diff(idx)
            cv = float(np.std(spac)/(np.mean(spac)+1e-12))
            feats["peak_spacing_mean"] = float(np.mean(spac))
            feats["peak_spacing_cv"]   = cv
            feats["peak_regularity"]   = float(1.0/(1.0+cv))
        else:
            feats["peak_spacing_mean"] = 0.0
            feats["peak_spacing_cv"]   = 1.0
            feats["peak_regularity"]   = 0.0
        if len(prom)>0:
            prom = np.array(prom, dtype=float)
            feats["peak_prom_median"] = float(np.median(prom))
            feats["peak_prom_cv"]     = float(prom.std(ddof=1)/(prom.mean()+1e-12)) if prom.size>1 else 0.0
        else:
            feats["peak_prom_median"]=0.0; feats["peak_prom_cv"]=0.0
        feats.update(_fft_features(yp))
    else:
        for k in ["peak_count_per100","peak_spacing_mean","peak_spacing_cv","peak_regularity",
                  "peak_prom_median","peak_prom_cv","dom_freq","spec_entropy","low_high_ratio","flatness"]:
            feats[k]=0.0

    if t.size:
        mag = np.abs(t); thr = np.quantile(mag, 0.5); big = mag >= thr
        sgn = np.sign(t) * big
        flips = np.sum((sgn[:-1]*sgn[1:])<0)
        denom = int(np.sum(big)) - 1
        feats["oscillation_rate"] = float(flips/denom) if denom>0 else 0.0
    else:
        feats["oscillation_rate"] = 0.0

    if a.size:
        B = 24
        hist,_ = np.histogram((a+np.pi)/(2*np.pi), bins=B, range=(0,1)); p = hist/(hist.sum()+1e-12)
        feats["dir_entropy"] = float(-(p[p>0]*np.log(p[p>0])).sum()/np.log(B))
        feats["axial_R2"] = float(np.abs(np.mean(np.exp(2j*a))))
        feats["quad_R4"]  = float(np.abs(np.mean(np.exp(4j*a))))
        dt = _turns(x,y)
        feats["ac2_turns"] = float(np.corrcoef(dt[:-2], dt[2:])[0,1]) if dt.size>2 and np.std(dt[:-2])>1e-12 and np.std(dt[2:])>1e-12 else 0.0
        st = np.vstack([np.diff(x), np.diff(y)]).T
        _, area2, _ = _convex_hull(st)
        r = np.hypot(st[:,0], st[:,1]); r95 = np.quantile(r, 0.95) if r.size else 0.0
        denom = np.pi*r95*r95 if r95>0 else 1.0
        feats["polar_hull_ratio"] = float(area2/denom)
    else:
        for k in ["dir_entropy","axial_R2","quad_R4","ac2_turns","polar_hull_ratio"]:
            feats[k]=0.0

    span = max(np.max(x)-np.min(x), np.max(y)-np.min(y))
    P = np.vstack([x,y]).T
    def _RDP(points, eps):
        if len(points) < 3: return points
        def dpt(p,a,b):
            if np.allclose(a,b): return np.hypot(*(p-a))
            t = np.clip(np.dot(p-a,b-a)/np.dot(b-a,b-a), 0, 1)
            proj = a + t*(b-a)
            return np.hypot(*(p-proj))
        stack=[(0,len(points)-1)]; keep=np.zeros(len(points), dtype=bool); keep[0]=keep[-1]=True
        while stack:
            i,j = stack.pop()
            a,b = points[i], points[j]
            dmax=0.0; imax=None
            for k in range(i+1, j):
                d = dpt(points[k], a, b)
                if d > dmax: dmax=d; imax=k
            if dmax > eps and imax is not None:
                keep[imax]=True; stack.append((i, imax)); stack.append((imax, j))
        return points[keep]
    for eps_pct in (0.01, 0.02):
        eps = eps_pct * (span if span>0 else 1.0)
        simp = _RDP(P, eps)
        Lsimp = float(np.sum(np.hypot(np.diff(simp[:,0]), np.diff(simp[:,1])))) if len(simp)>1 else 0.0
        L = _path_len(x, y)
        feats[f"rdp_retention_{int(eps_pct*100)}pct"] = float(Lsimp/(L+1e-12))

    rtol = 0.01 * (span if span>0 else 1.0)
    rec = 0; total = 0
    for i in range(2, n):
        d = np.hypot(x[:i]-x[i], y[:i]-y[i])
        if np.any(d <= rtol): rec += 1
        total += 1
    feats["recurrence_ratio"] = float(rec/max(1,total))

    gx=10; gy=10
    if span>0:
        xi = np.floor((np.array(x)-np.min(x))/max(1e-12, (np.max(x)-np.min(x))) * (gx-1)).astype(int)
        yi = np.floor((np.array(y)-np.min(y))/max(1e-12, (np.max(y)-np.min(y))) * (gy-1)).astype(int)
    else:
        xi=np.zeros(n,int); yi=np.zeros(n,int)
    grid = np.zeros((gx,gy), dtype=int)
    for i in range(n): grid[xi[i], yi[i]] += 1
    p = grid.ravel().astype(float); p = p/p.sum() if p.sum()>0 else p
    feats["occupancy_entropy"] = float(-(p[p>0]*np.log(p[p>0])).sum()/np.log(gx*gy)) if p.sum()>0 else 0.0
    visited=set(); half=max(1,int((grid>0).sum()/2)); t_half=0
    for i in range(n):
        visited.add((xi[i],yi[i]))
        if len(visited) >= half: t_half=i+1; break
    feats["time_to_half_coverage"] = float(t_half)

    _, rmse1 = _polyfit_r2_rmse(x, y, 1)
    feats["poly_rmse_deg1"] = float(rmse1 if np.isfinite(rmse1) else 0.0)

    return {k: float(feats.get(k, 0.0)) for k in KEEP_FEATURES}

# -------- worker (top-level for Windows pickling) --------
def worker_job(job):
    csv_path, init_state, rule = job
    try:
        x, y = read_xy(csv_path)
        if x is None:
            return None, dict(file=csv_path, error="missing_or_invalid_xy")
        feats = compute_features_for_series(x, y)
        feats.update(dict(
            file=csv_path,
            rule=int(rule),
            initial_state=str(init_state),
        ))
        return feats, None
    except Exception as e:
        return None, dict(file=csv_path, error=str(e))

# -------- scanning utilities --------
def scan_all_csvs(data_root: str):
    """
    Yields tuples: (csv_path, init_state_folder, rule_id)
    Expects: <data_root>\\<INIT>_100\\original\\dfOriginal_r{rule}.csv
    """
    root = Path(data_root)
    pat_dir = re.compile(r"^[01]{7}_100$")
    pat_file = re.compile(r"dfOriginal_r(\d+)\.csv$", re.IGNORECASE)
    for sub in root.iterdir():
        if not sub.is_dir(): continue
        if not pat_dir.match(sub.name): continue
        init_state = sub.name.split("_")[0]
        orig = sub / "original"
        if not orig.exists(): continue
        for f in orig.iterdir():
            if not f.is_file(): continue
            m = pat_file.match(f.name)
            if not m: continue
            rule = int(m.group(1))
            yield str(f), init_state, rule

def safe_load_features(features_json_path: str):
    """Load feature order from features.json (list or {'features': [...]})"""
    with open(features_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    feats = obj["features"] if isinstance(obj, dict) and "features" in obj else obj
    if not isinstance(feats, list):
        raise ValueError("features.json must be a list or a dict with key 'features'.")
    # sanity vs whitelist (warn only)
    missing = [k for k in KEEP_FEATURES if k not in feats]
    extras  = [k for k in feats if k not in KEEP_FEATURES]
    if missing:
        print(f"[WARN] features.json missing expected features: {missing}")
    if extras:
        print(f"[WARN] features.json contains non-whitelisted extras: {extras}")
    return feats

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Predict movement types for original/no-barrier CAR trajectories.")    
    ap.add_argument("--data_root", required=True, help="Root folder containing <initial_state>_<num_gen>/original/dfOriginal_r{rule}.csv")
    ap.add_argument("--model_dir", required=True, help=r'Folder with scaler.pkl, model.pkl, features.json, label_names.json')
    ap.add_argument("--out_dir", required=True, help=r'Where to save features + predictions')
    ap.add_argument("--workers", type=int, default=1, help="Parallel feature jobs (Windows-safe default = 1)")
    ap.add_argument("--random_state", type=int, default=42, help="For manifest only (not used)")
    args = ap.parse_args()

    print("\n===== ORIGINAL MOVEMENT-TYPE PREDICTION =====")
    print(f"Data root  : {args.data_root}")
    print(f"Model dir  : {args.model_dir}")
    print(f"Output dir : {args.out_dir}")
    print(f"Workers    : {args.workers}")
    print("============================================\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # artifacts
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    model_path  = os.path.join(args.model_dir, "model.pkl")
    feats_path  = os.path.join(args.model_dir, "features.json")
    labels_path = os.path.join(args.model_dir, "label_names.json")
    for p in [scaler_path, model_path, feats_path, labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)
    feat_order = safe_load_features(feats_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)
    if isinstance(label_names, dict) and "labels" in label_names:
        label_names = label_names["labels"]
    classes = list(getattr(model, "classes_", label_names))

    # jobs
    jobs = list(scan_all_csvs(args.data_root))
    if not jobs:
        raise RuntimeError("No trajectory CSVs found under --data_root.")
    print(f"Found {len(jobs)} trajectories.")

    rows, errs = [], []

    # process (sequential if workers==1 to avoid any Windows pickling headaches)
    if args.workers <= 1:
        if tqdm is not None:
            for j in tqdm(jobs, desc="Computing features", unit="traj", dynamic_ncols=True):
                ok, er = worker_job(j)
                if ok is not None: rows.append(ok)
                if er is not None: errs.append(er)
        else:
            for j in jobs:
                ok, er = worker_job(j)
                if ok is not None: rows.append(ok)
                if er is not None: errs.append(er)
    else:
        if tqdm is not None:
            pbar = tqdm(total=len(jobs), desc="Computing features", unit="traj", dynamic_ncols=True)
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(worker_job, j) for j in jobs]
            for fut in as_completed(futs):
                res = fut.result()
                if res is None: continue
                ok, er = res
                if ok is not None: rows.append(ok)
                if er is not None: errs.append(er)
                if tqdm is not None: pbar.update(1)
        if tqdm is not None: pbar.close()

    if not rows:
        raise RuntimeError("No features computed; see errors.csv for details.")

    # raw features
    df_raw = pd.DataFrame(rows)
    for k in KEEP_FEATURES:
        if k not in df_raw.columns:
            df_raw[k] = 0.0
    meta_cols = ["file", "rule", "initial_state"]
    feat_cols = [c for c in feat_order if c in KEEP_FEATURES]
    df_raw = df_raw[meta_cols + feat_cols]
    raw_path = os.path.join(args.out_dir, "full_features_raw.csv")
    df_raw.to_csv(raw_path, index=False)

    # normalized features
    X = df_raw[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0).values
    Xz = scaler.transform(X)
    df_norm = pd.DataFrame(Xz, columns=feat_cols)
    for c in reversed(meta_cols):
        df_norm.insert(0, c, df_raw[c].values)
    norm_path = os.path.join(args.out_dir, "full_features_norm.csv")
    df_norm.to_csv(norm_path, index=False)

    # predict
    proba = model.predict_proba(Xz)
    df_pred = df_raw[meta_cols].copy()
    for i, cls in enumerate(classes):
        df_pred[str(cls)] = proba[:, i]
    pred_idx = np.argmax(proba, axis=1)
    df_pred["pred"] = [classes[i] for i in pred_idx]
    df_pred["pred_conf"] = proba.max(axis=1)

    pred_path = os.path.join(args.out_dir, "full_predictions.csv")
    df_pred.to_csv(pred_path, index=False)

    # counts
    counts = df_pred["pred"].value_counts().rename_axis("label").reset_index(name="count")
    counts_path = os.path.join(args.out_dir, "full_pred_counts.csv")
    counts.to_csv(counts_path, index=False)

    # errors
    if errs:
        pd.DataFrame(errs).to_csv(os.path.join(args.out_dir, "errors.csv"), index=False)

    # manifest
    manifest = {
        "script": "predict_movement_og.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {
            "data_root": args.data_root,
            "model_dir": args.model_dir,
            "out_dir": args.out_dir,
            "workers": int(args.workers),
            "random_state": int(args.random_state),
        },
        "n_inputs": len(jobs),
        "n_rows_ok": int(len(df_raw)),
        "n_errors": int(len(errs)),
        "feature_order_used": feat_cols,
        "whitelist_len": len(KEEP_FEATURES),
        "label_names": classes,
        "artifacts_in": {
            "scaler.pkl": scaler_path,
            "model.pkl": model_path,
            "features.json": feats_path,
            "label_names.json": labels_path
        },
        "artifacts_out": {
            "full_features_raw.csv": raw_path,
            "full_features_norm.csv": norm_path,
            "full_predictions.csv": pred_path,
            "full_pred_counts.csv": counts_path,
            "errors.csv": os.path.join(args.out_dir, "errors.csv") if errs else None
        }
    }
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {norm_path}")
    print(f"  {pred_path}")
    print(f"  {counts_path}")
    if errs:
        print(f"  {os.path.join(args.out_dir, 'errors.csv')}  (n={len(errs)})")
    print("\nDone.")

if __name__ == "__main__":
    main()
