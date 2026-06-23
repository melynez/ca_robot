import os, re, math, json, time, argparse, warnings, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ------------------------
# Fixed 40-feature whitelist (order matters)
# ------------------------
KEEP_FEATURES = [
    "ac1_step","ac2_turns","axial_R2","axis_adherence_15deg","cv_step","dir_entropy","dom_freq",
    "flatness","hull_area_norm","hull_circularity","kurt_turn","lin_corner_changes","linear_r2",
    "low_high_ratio","mad_step","max_abs_turn","mean_turn","occupancy_entropy","orientation_eff",
    "oscillation_rate","pc_var_ratio","peak_count_per100","peak_prom_cv","peak_prom_median",
    "peak_regularity","peak_spacing_cv","peak_spacing_mean","polar_hull_ratio","poly_rmse_deg1",
    "quad_R4","rdp_retention_1pct","rdp_retention_2pct","recurrence_ratio","run_len_mean","skew_turn",
    "spec_entropy","std_turn","straightness","time_to_half_coverage","turn_consistency"
]

# ------------------------
# Robust CSV reader for Final X/Y (case/space tolerant)
# ------------------------
def read_xy(csv_path: str):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    cols = {c.lower().replace(" ","_"): c for c in df.columns}
    cx = cols.get("final_x") or next((c for k,c in cols.items() if "final_x" in k), None)
    cy = cols.get("final_y") or next((c for k,c in cols.items() if "final_y" in k), None)
    if cx is None:
        xs = [c for k,c in cols.items() if k.endswith("_x")]
        if xs: cx = xs[-1]
    if cy is None:
        ys = [c for k,c in cols.items() if k.endswith("_y")]
        if ys: cy = ys[-1]
    if cx is None or cy is None:
        return None, None
    x = pd.to_numeric(df[cx], errors="coerce").to_numpy()
    y = pd.to_numeric(df[cy], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or x.size != y.size:
        return None, None
    return x, y

# ------------------------
# Math / metrics helpers (identical to training)
# ------------------------
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

# ------------------------
# Feature computation (returns ONLY the keep-list)
# ------------------------
def compute_features_for_series(x, y):
    feats = {}
    n = len(x)
    if n < 3: return feats

    L = _path_len(x,y); D = _net_disp(x,y)
    a = _headings(x,y); t = _turns(x,y); at = np.abs(t)

    # straightness + turn stats
    feats["straightness"] = float(D/(L+1e-12))
    feats["mean_turn"] = float(at.mean()) if at.size else 0.0
    feats["std_turn"]  = float(at.std(ddof=1)) if at.size>1 else 0.0
    feats["skew_turn"] = float(((at - at.mean())**3).mean() / (at.std()+1e-12)**3) if at.size>2 and at.std()>1e-12 else 0.0
    feats["kurt_turn"] = float(((at - at.mean())**4).mean() / (at.var()+1e-12)**2) if at.size>3 and at.var()>1e-12 else 0.0

    # orientation efficiency
    if a.size:
        C = np.mean(np.cos(a)); S = np.mean(np.sin(a))
        mean_dir = math.atan2(S, C)
        feats["orientation_eff"] = float(D / (np.sum(np.abs(_wrap_pi(a-mean_dir)))+1e-9))
    else:
        feats["orientation_eff"] = 0.0

    # turn consistency + run lengths
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

    # step series
    dx = np.diff(x); dy = np.diff(y); s = np.hypot(dx,dy)
    feats["cv_step"]  = float(s.std(ddof=1)/(s.mean()+1e-12)) if s.size>1 else 0.0
    feats["mad_step"] = float(np.median(np.abs(s-np.median(s)))) if s.size else 0.0
    feats["ac1_step"] = float(np.corrcoef(s[:-1], s[1:])[0,1]) if s.size>2 and np.std(s[:-1])>1e-12 and np.std(s[1:])>1e-12 else 0.0

    # PCA + hull
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

    # detrended-perp signal features + FFT
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

    # oscillation rate in turns
    t = _turns(x,y)
    if t.size:
        mag = np.abs(t); thr = np.quantile(mag, 0.5); big = mag >= thr
        sgn = np.sign(t) * big
        flips = np.sum((sgn[:-1]*sgn[1:])<0)
        denom = int(np.sum(big)) - 1
        feats["oscillation_rate"] = float(flips/denom) if denom>0 else 0.0
    else:
        feats["oscillation_rate"] = 0.0

    # directional stats & velocity-space hull
    a = _headings(x,y)
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

    # RDP retention
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

    # recurrence
    rtol = 0.01 * (span if span>0 else 1.0)
    rec = 0; total = 0
    for i in range(2, n):
        d = np.hypot(x[:i]-x[i], y[:i]-y[i])
        if np.any(d <= rtol): rec += 1
        total += 1
    feats["recurrence_ratio"] = float(rec/max(1,total))

    # occupancy + coverage time
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

    # polynomial fit roughness
    _, rmse1 = _polyfit_r2_rmse(x, y, 1)
    feats["poly_rmse_deg1"] = float(rmse1 if np.isfinite(rmse1) else 0.0)

    return {k: float(feats.get(k, 0.0)) for k in KEEP_FEATURES}

# ------------------------
# Loop detection: immediate/pair/triplet repeats
# ------------------------
def loop_score_segment(x, y):
    """Return loop score in {0.0, 0.25, 0.5, 0.75}."""
    n = len(x)
    if n <= 1: return 0.0
    rep = pair = trip = 0
    for i in range(1, n):
        if np.isclose(x[i], x[i-1]) and np.isclose(y[i], y[i-1]):
            rep += 1
        elif i > 1 and np.isclose(x[i], x[i-2]) and np.isclose(y[i], y[i-2]):
            pair += 1
        elif i > 2 and np.isclose(x[i], x[i-3]) and np.isclose(y[i], y[i-3]):
            trip += 1
    repeat_pct = 100.0 * rep / n
    pair_pct   = 100.0 * pair / n
    trip_pct   = 100.0 * trip / n
    if repeat_pct > 70: return 0.75
    if pair_pct   > 70: return 0.5
    if trip_pct   > 70: return 0.25
    return 0.0

# ------------------------
# Divergence metrics for pre/post distributions
# ------------------------
def js_divergence(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p /= (p.sum() + 1e-12); q /= (q.sum() + 1e-12)
    m = 0.5*(p+q)
    def _kl(a,b):
        mask = (a>0) & (b>0)
        return np.sum(a[mask]*np.log(a[mask]/b[mask]))
    js = 0.5*_kl(p,m) + 0.5*_kl(q,m)
    return float(js/np.log(2))  # bits

def l1_half(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p /= (p.sum() + 1e-12); q /= (q.sum() + 1e-12)
    return float(0.5*np.abs(p-q).sum())

# ------------------------
# Barrier file discovery (…\<init>_100\<barrier_dir_name>\dfBarrier_r*.csv)
# ------------------------
def find_barrier_csvs(root_dir, barrier_dir_name="gen50_size20_angle90_extra0"):
    out = []
    for dirpath, _, files in os.walk(root_dir):
        parts = dirpath.replace("\\","/").split("/")
        if parts and parts[-1].lower() == barrier_dir_name.lower():
            for f in files:
                fl = f.lower()
                if fl.endswith(".csv") and fl.startswith("dfbarrier_r"):
                    out.append(os.path.join(dirpath, f))
    return sorted(out)

def parse_init_and_rule_from_path(path):
    # ...\<init>_100\<barrier>\dfBarrier_rXXX.csv
    init_state = "unknown"
    norm = path.replace("\\","/")
    for seg in norm.split("/"):
        m = re.match(r"^([01]{3,64})_100$", seg)
        if m:
            init_state = m.group(1).zfill(7)
            break
    m = re.search(r"dfBarrier[_\-]?r(\d+)\.csv$", os.path.basename(path), flags=re.I)
    rule = int(m.group(1)) if m else None
    return init_state, rule

# ------------------------
# Worker: slice, filter, features (Windows-safe)
# ------------------------
def worker_barrier(job):
    """
    job = (csv_path, min_total_skip, loop_threshold)
    Returns dict with meta + before_feats + after_feats (or flags).
    """
    csv_path, min_total_skip, loop_threshold = job
    init_state, rule = parse_init_and_rule_from_path(csv_path)

    x, y = read_xy(csv_path)
    if x is None:
        return dict(meta=dict(file=csv_path, initial_state=init_state, rule=rule,
                              n_total=0, status="read_fail"))

    n = len(x)
    status = "OK"
    if n < min_total_skip:
        status = "SKIP_LT75"
    elif 75 <= n < 100:
        status = "PARTIAL_75_99"
    else:
        status = "FULL_GE100"

    # If too short, skip everything but keep the meta
    if status == "SKIP_LT75":
        return dict(meta=dict(file=csv_path, initial_state=init_state, rule=rule,
                              n_total=n, n_before=0, n_after=0,
                              status=status, loop=False, loop_score=0.0),
                    before=None, after=None)

    # Slices: first 50 and last 50 (may overlap if n<100)
    b0, b1 = 0, min(50, n)
    a0, a1 = max(0, n-50), n
    xb, yb = x[b0:b1], y[b0:b1]
    xa, ya = x[a0:a1], y[a0:a1]

    # Loop detection on AFTER
    loop_s = loop_score_segment(xa, ya) if len(xa) >= 3 else 0.0
    is_loop = (loop_s > loop_threshold)

    before_feats = compute_features_for_series(xb, yb) if len(xb) >= 3 else None
    after_feats  = None if is_loop else (compute_features_for_series(xa, ya) if len(xa) >= 3 else None)

    meta = dict(file=csv_path, initial_state=init_state, rule=rule,
                n_total=n, n_before=len(xb), n_after=len(xa),
                status=status, overlap=(n<100), loop=is_loop, loop_score=loop_s)

    return dict(meta=meta, before=before_feats, after=after_feats)

# ------------------------
# Utility helpers
# ------------------------
def sanitize_prob_col(label: str) -> str:
    return f"prob_{str(label).replace(' ', '_')}"

def md5_of_file(path: str) -> str:
    if not os.path.exists(path): return ""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Predict movement types before and after barrier interaction."
    )
    ap.add_argument("--data_root", required=True, help=r'Path to 7bit_multiverse root')
    ap.add_argument("--model_dir", required=True, help=r'Folder with scaler.pkl, model.pkl, features.json, label_names.json')
    ap.add_argument("--out_dir",   required=True, help=r'Where to save outputs')
    ap.add_argument("--barrier_dir_name", default="gen50_size20_angle90_extra0")
    ap.add_argument("--loop_threshold", type=float, default=0.1, help="Loop score > threshold => loop (0/0.25/0.5/0.75)")
    ap.add_argument("--min_total_points_skip", type=int, default=75, help="Skip entirely if total points < this")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()


    print("\n===== BARRIER MOVEMENT-TYPE PREDICTION =====")
    print(f"Data root  : {args.data_root}")
    print(f"Model dir  : {args.model_dir}")
    print(f"Output dir : {args.out_dir}")
    print(f"Barrier dir: {args.barrier_dir_name}")
    print(f"Workers    : {args.workers}")
    print("===========================================\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load artifacts
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    model_path  = os.path.join(args.model_dir, "model.pkl")
    feats_path  = os.path.join(args.model_dir, "features.json")
    labels_path = os.path.join(args.model_dir, "label_names.json")
    for p in [scaler_path, model_path, feats_path, labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)
    with open(feats_path, "r", encoding="utf-8") as f:
        feat_order = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)
    if isinstance(label_names, dict) and "labels" in label_names:
        label_names = label_names["labels"]
    classes = list(getattr(model, "classes_", label_names))

    # Find barrier CSVs
    files = find_barrier_csvs(args.data_root, args.barrier_dir_name)
    if not files:
        print(f"No barrier CSVs found under '{args.barrier_dir_name}'.")
        return
    print(f"Found {len(files)} barrier trajectories.")

    # Feature extraction
    jobs = [(p, args.min_total_points_skip, args.loop_threshold) for p in files]
    results = []
    if args.workers <= 1:
        it = tqdm(jobs, desc="Feature extraction", unit="file", dynamic_ncols=True) if tqdm else jobs
        for j in it:
            results.append(worker_barrier(j))
    else:
        pbar = tqdm(total=len(jobs), desc="Feature extraction", unit="file", dynamic_ncols=True) if tqdm else None
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(worker_barrier, j) for j in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())
                if pbar is not None: pbar.update(1)
        if pbar is not None: pbar.close()

    if not results:
        print("Nothing processed.")
        return

    # Build per-segment rows for prediction
    seg_rows = []
    meta_rows = []
    for r in results:
        m = r["meta"]
        meta_rows.append(m)
        file = m["file"]; init_state = m["initial_state"]; rule = m["rule"]
        if r["before"] is not None:
            row_b = {"file": file, "initial_state": init_state, "rule": rule, "segment": "before"}
            row_b.update(r["before"])
            seg_rows.append(row_b)
        if r["after"] is not None:
            row_a = {"file": file, "initial_state": init_state, "rule": rule, "segment": "after"}
            row_a.update(r["after"])
            seg_rows.append(row_a)

    meta_df = pd.DataFrame(meta_rows)
    seg_df  = pd.DataFrame(seg_rows)

    # Predict where features exist
    pred_df = pd.DataFrame()
    if not seg_df.empty:
        # Ensure all 40 features & correct order
        for c in feat_order:
            if c not in seg_df.columns:
                seg_df[c] = 0.0
        X  = seg_df[feat_order].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        Xz = scaler.transform(X)
        proba = model.predict_proba(Xz)
        yhat  = model.classes_[np.argmax(proba, axis=1)]
        maxp  = proba.max(axis=1)
        # entropy & margin
        ent = []
        mar = []
        for i in range(proba.shape[0]):
            p = proba[i].clip(1e-12, 1.0)
            p = p / p.sum()
            ent.append(float(-(p*np.log(p)).sum()))
            s = np.sort(p)[::-1]
            mar.append(float(s[0] - (s[1] if len(s)>1 else 0.0)))
        pred_df = seg_df[["file","initial_state","rule","segment"]].copy()
        for i, cls in enumerate(model.classes_):
            pred_df[sanitize_prob_col(cls)] = proba[:, i]
        pred_df["pred_label"]    = yhat
        pred_df["pred_max_prob"] = maxp
        pred_df["pred_entropy"]  = ent
        pred_df["pred_margin"]   = mar

    # Save per-segment predictions
    out_seg = os.path.join(args.out_dir, "barrier_predictions_per_segment.csv")
    if not pred_df.empty:
        pred_df.to_csv(out_seg, index=False)

    # Compose per-rule summary (compare before vs after)
    def _lookup(file, rule, segment):
        if pred_df.empty: return None
        rows = pred_df[(pred_df.file==file) & (pred_df.rule==rule) & (pred_df.segment==segment)]
        return None if rows.empty else rows.iloc[0]

    summary_rows = []
    for _, m in meta_df.iterrows():
        file = m["file"]; init_state = m["initial_state"]; rule = m["rule"]
        n_total = int(m.get("n_total", 0)); n_before = int(m.get("n_before", 0)); n_after = int(m.get("n_after", 0))
        status = m.get("status", "OK")
        loop  = bool(m.get("loop", False))
        loop_score = float(m.get("loop_score", 0.0))
        overlap = bool(m.get("overlap", False))

        rb = _lookup(file, rule, "before")
        ra = _lookup(file, rule, "after")

        s = dict(
            file=file, initial_state=init_state, rule=rule,
            n_total=n_total, n_before=n_before, n_after=n_after,
            status=status, overlap=overlap, loop=loop, loop_score=loop_score,

            pred_before=None, conf_before=None, entropy_before=None, margin_before=None,
            pred_after=None,  conf_after=None,  entropy_after=None,  margin_after=None,

            changed=None,
            orig_label=None, orig_label_prob_before=None, orig_label_prob_after=None,
            new_label=None,  new_label_prob_after=None,
            delta_orig_label_prob=None,
            delta_max_prob=None,
            delta_entropy=None,
            delta_margin=None,
            js_divergence=None,
            l1_dist=None,
        )

        if rb is not None:
            s["pred_before"]    = str(rb["pred_label"])
            s["conf_before"]    = float(rb["pred_max_prob"])
            s["entropy_before"] = float(rb["pred_entropy"])
            s["margin_before"]  = float(rb["pred_margin"])
            p_before = np.array([rb[sanitize_prob_col(c)] for c in classes], float)

        if ra is not None:
            s["pred_after"]     = str(ra["pred_label"])
            s["conf_after"]     = float(ra["pred_max_prob"])
            s["entropy_after"]  = float(ra["pred_entropy"])
            s["margin_after"]   = float(ra["pred_margin"])
            p_after  = np.array([ra[sanitize_prob_col(c)] for c in classes], float)

        # deltas & divergences if both sides exist
        if rb is not None and ra is not None:
            s["changed"] = (s["pred_before"] != s["pred_after"])
            s["delta_max_prob"] = float(s["conf_after"] - s["conf_before"])
            s["delta_entropy"]  = float(s["entropy_after"] - s["entropy_before"])
            s["delta_margin"]   = float(s["margin_after"] - s["margin_before"])
            s["js_divergence"]  = js_divergence(p_before, p_after)
            s["l1_dist"]        = l1_half(p_before, p_after)

            # strength of original category before vs after
            idx_orig = classes.index(s["pred_before"])
            s["orig_label"]              = s["pred_before"]
            s["orig_label_prob_before"]  = float(p_before[idx_orig])
            s["orig_label_prob_after"]   = float(p_after[idx_orig])
            s["delta_orig_label_prob"]   = float(p_after[idx_orig] - p_before[idx_orig])

            # if changed, strength of new category
            if s["changed"]:
                idx_new = classes.index(s["pred_after"])
                s["new_label"]            = s["pred_after"]
                s["new_label_prob_after"] = float(p_after[idx_new])

        summary_rows.append(s)

    summary_df = pd.DataFrame(summary_rows)
    out_sum = os.path.join(args.out_dir, "barrier_change_summary.csv")
    summary_df.to_csv(out_sum, index=False)

    # Transition matrices (only where both before & after predictions exist and not loop/skip)
    valid = summary_df[
        summary_df["pred_before"].notna() &
        summary_df["pred_after"].notna() &
        (summary_df["status"] != "SKIP_LT75") &
        (~summary_df["loop"])
    ].copy()

    if not valid.empty:
        # overall
        cm = pd.crosstab(valid["pred_before"], valid["pred_after"]).sort_index(axis=0).sort_index(axis=1)
        cm.to_csv(os.path.join(args.out_dir, "barrier_transition_matrix.csv"))
        cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
        cm_norm.to_csv(os.path.join(args.out_dir, "barrier_transition_matrix_row_norm.csv"))

        # per-initial-state tall-format counts
        trans_rows = valid.groupby(["initial_state","pred_before","pred_after"]).size().reset_index(name="count")
        trans_rows.to_csv(os.path.join(args.out_dir, "barrier_transition_counts_by_initial_state.csv"), index=False)

    # Class/count summaries
    counts_overall = valid["pred_after"].value_counts().rename_axis("label").reset_index(name="count")
    counts_overall.to_csv(os.path.join(args.out_dir, "barrier_pred_after_counts_overall.csv"), index=False)

    if not valid.empty:
        by_is = valid.groupby(["initial_state","pred_after"]).size().reset_index(name="count")
        by_is.rename(columns={"pred_after":"label"}, inplace=True)
        by_is.to_csv(os.path.join(args.out_dir, "barrier_pred_after_counts_by_initial_state.csv"), index=False)

    # Quick stats (overall & per-initial-state)
    def _stats(df):
        if df.empty: 
            return dict(n=0, changed_rate=np.nan, mean_js=np.nan, mean_l1=np.nan,
                        mean_delta_max_prob=np.nan, mean_delta_entropy=np.nan, mean_delta_margin=np.nan)
        return dict(
            n=len(df),
            changed_rate=float(np.mean(df["changed"].astype(bool))),
            mean_js=float(np.nanmean(df["js_divergence"].astype(float))),
            mean_l1=float(np.nanmean(df["l1_dist"].astype(float))),
            mean_delta_max_prob=float(np.nanmean(df["delta_max_prob"].astype(float))),
            mean_delta_entropy=float(np.nanmean(df["delta_entropy"].astype(float))),
            mean_delta_margin=float(np.nanmean(df["delta_margin"].astype(float))),
        )

    stats_overall = _stats(valid)
    pd.DataFrame([stats_overall]).to_csv(os.path.join(args.out_dir, "barrier_stats_overall.csv"), index=False)

    stats_by_is = []
    for is_code, sub in valid.groupby("initial_state"):
        d = _stats(sub); d["initial_state"] = is_code
        stats_by_is.append(d)
    pd.DataFrame(stats_by_is).to_csv(os.path.join(args.out_dir, "barrier_stats_by_initial_state.csv"), index=False)

    # Manifest with checksums of core CSVs
    core = {
        "barrier_predictions_per_segment.csv": out_seg,
        "barrier_change_summary.csv": out_sum,
        "barrier_transition_matrix.csv": os.path.join(args.out_dir, "barrier_transition_matrix.csv"),
        "barrier_transition_matrix_row_norm.csv": os.path.join(args.out_dir, "barrier_transition_matrix_row_norm.csv"),
        "barrier_pred_after_counts_overall.csv": os.path.join(args.out_dir, "barrier_pred_after_counts_overall.csv"),
        "barrier_pred_after_counts_by_initial_state.csv": os.path.join(args.out_dir, "barrier_pred_after_counts_by_initial_state.csv"),
        "barrier_stats_overall.csv": os.path.join(args.out_dir, "barrier_stats_overall.csv"),
        "barrier_stats_by_initial_state.csv": os.path.join(args.out_dir, "barrier_stats_by_initial_state.csv"),
    }
    manifest = {
        "script": "predict_movement_barrier.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {
            "data_root": args.data_root,
            "model_dir": args.model_dir,
            "out_dir": args.out_dir,
            "barrier_dir_name": args.barrier_dir_name,
            "loop_threshold": float(args.loop_threshold),
            "min_total_points_skip": int(args.min_total_points_skip),
            "workers": int(args.workers),
        },
        "model_artifacts": {
            "scaler.pkl": scaler_path,
            "model.pkl": model_path,
            "features.json": feats_path,
            "label_names.json": labels_path
        },
        "classes": classes,
        "feature_order": feat_order,
        "n_inputs": len(files),
        "n_processed_segments": 0 if pred_df.empty else int(len(pred_df)),
        "n_rules_valid": int(valid.shape[0]),
        "counts": {
            "n_skip_lt75": int((meta_df["status"]=="SKIP_LT75").sum()),
            "n_partial_75_99": int((meta_df["status"]=="PARTIAL_75_99").sum()),
            "n_full_ge100": int((meta_df["status"]=="FULL_GE100").sum()),
            "n_loops": int(meta_df["loop"].sum()),
        },
        "checksums_md5": {k: md5_of_file(v) for k,v in core.items() if os.path.exists(v)}
    }
    with open(os.path.join(args.out_dir, "barrier_run.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved:")
    if os.path.exists(out_seg): print(" ", out_seg)
    print(" ", out_sum)
    for k,v in core.items():
        if os.path.exists(v): print(" ", v)
    print(" ", os.path.join(args.out_dir, "barrier_run.json"))
    print("Done.")

if __name__ == "__main__":
    main()






