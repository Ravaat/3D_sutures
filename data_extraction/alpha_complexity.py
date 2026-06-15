import os
import re
import json
import numpy as np
import pandas as pd
import alphashape
import trimesh
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures


np.random.seed(42)


# -----------------------------------------------------
# Sampling
# -----------------------------------------------------

def sample_surface(mesh, n=2000):
    try:
        return mesh.sample(n)
    except:
        return np.empty((0,3))


# -----------------------------------------------------
# Nearest neighbour scale
# -----------------------------------------------------

def avg_nn(coords, k=20):

    if len(coords) < 2:
        return np.nan

    k = min(k, len(coords)-1)

    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=k+1)
    dists = dists[:,1:]

    return np.mean(dists)


# -----------------------------------------------------
# Alpha volume computation
# -----------------------------------------------------

def alpha_volume(coords, alpha):

    try:

        shape = alphashape.alphashape(coords, alpha)

        if shape is None:
            return np.nan

        if not hasattr(shape, "volume"):
            return np.nan

        vol = shape.volume

        if vol <= 0:
            return np.nan

        return vol

    except:
        return np.nan


# -----------------------------------------------------
# Binary root search for alpha
# -----------------------------------------------------

def solve_alpha_binary(coords, real_vol, alpha_min, alpha_max, tol=0.01, max_iter=25):

    v_low = alpha_volume(coords, alpha_min)
    v_high = alpha_volume(coords, alpha_max)

    if not np.isfinite(v_low) or not np.isfinite(v_high):
        return None

    if (v_low - real_vol) * (v_high - real_vol) > 0:
        return None

    a_low = alpha_min
    a_high = alpha_max

    for _ in range(max_iter):

        a_mid = np.sqrt(a_low * a_high)
        v_mid = alpha_volume(coords, a_mid)

        if not np.isfinite(v_mid):
            return None

        if abs(v_mid - real_vol) / real_vol < tol:
            return a_mid, v_mid

        if (v_low - real_vol) * (v_mid - real_vol) < 0:
            a_high = a_mid
            v_high = v_mid
        else:
            a_low = a_mid
            v_low = v_mid

    return a_mid, v_mid


# -----------------------------------------------------
# Sweep fallback
# -----------------------------------------------------

def sweep_alpha(coords, real_vol, scale):

    alphas = np.logspace(
        np.log10(scale * 0.05),
        np.log10(scale * 200),
        120
    )

    vols = np.array([alpha_volume(coords,a) for a in alphas])

    mask = np.isfinite(vols)

    if np.sum(mask) < 3:
        return None

    alphas = alphas[mask]
    vols = vols[mask]

    idx = np.argmin(np.abs((vols / real_vol) - 1))

    return alphas, vols, idx


# -----------------------------------------------------
# Exact alpha interpolation
# -----------------------------------------------------

def interpolate_alpha(alphas, vols, real_vol):

    diff = vols - real_vol

    sign = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign) == 0:
        return None

    i = sign[0]

    a1,a2 = alphas[i], alphas[i+1]
    v1,v2 = vols[i], vols[i+1]

    loga1,loga2 = np.log(a1), np.log(a2)

    t = (real_vol - v1) / (v2 - v1)

    return float(np.exp(loga1 + t*(loga2-loga1)))


# -----------------------------------------------------
# Suture grouping
# -----------------------------------------------------

def extract_suture_type(name):

    match = re.search(r'([^_]+_[^_]+_suture_[^_]+)', name)

    if match:
        return match.group(1)

    parts = name.split('_suture_')

    if len(parts)>1:
        after = parts[1].split('_')[0]
        return parts[0] + "_suture_" + after

    return name


# -----------------------------------------------------
# Mesh processing
# -----------------------------------------------------

def process_mesh(name, real_vol, mesh_dir, samples=2000):

    path = os.path.join(mesh_dir,f"{name}.ply")

    try:
        mesh = trimesh.load(path, force='mesh')
    except:
        return None

    coords = sample_surface(mesh,samples)

    if len(coords) < 50:
        return None

    if real_vol is None or np.isnan(real_vol):
        return None

    scale = avg_nn(coords)

    alpha_min = scale * 0.05
    alpha_max = scale * 200

    binary = solve_alpha_binary(coords, real_vol, alpha_min, alpha_max)

    if binary is not None:

        alpha_exact, vol_exact = binary

        return {
            "filename": name,
            "exact_alpha": float(alpha_exact),
            "best_alpha": float(alpha_exact),
            "best_volume": float(vol_exact),
            "actual_volume": float(real_vol),
            "volume_ratio": float(vol_exact/real_vol),
            "alphas": [],
            "volumes": []
        }

    sweep = sweep_alpha(coords, real_vol, scale)

    if sweep is None:
        return None

    alphas, vols, idx = sweep

    alpha_exact = interpolate_alpha(alphas, vols, real_vol)

    return {

        "filename": name,

        "best_alpha": float(alphas[idx]),
        "exact_alpha": alpha_exact,

        "best_volume": float(vols[idx]),
        "actual_volume": float(real_vol),

        "volume_ratio": float(vols[idx]/real_vol),

        "alphas": alphas.tolist(),
        "volumes": vols.tolist()
    }


# -----------------------------------------------------
# Pipeline
# -----------------------------------------------------

def run_pipeline(mesh_dir, volume_csv, n_jobs=None):

    print("Reading CSV")

    df = pd.read_csv(volume_csv)

    df['filename'] = (
        df['filename']
        .str.strip()
        .str.replace(".ply","",regex=False)
    )

    vol_map = dict(zip(df.filename, df.volume_mm3))

    meshes = [f for f in os.listdir(mesh_dir) if f.endswith(".ply")]
    names = [os.path.splitext(f)[0] for f in meshes]

    tasks = [(n,vol_map.get(n),mesh_dir) for n in names]

    summary=[]
    results={}

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:

        futures=[ex.submit(process_mesh,*t) for t in tasks]

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):

            r=f.result()

            if r is None:
                continue

            summary.append({

                "filename": r["filename"],
                "best_alpha": r["best_alpha"],
                "exact_alpha": r["exact_alpha"],
                "best_volume": r["best_volume"],
                "actual_volume": r["actual_volume"],
                "volume_ratio": r["volume_ratio"]

            })

            results[r["filename"]] = {
                "alphas": r["alphas"],
                "volumes": r["volumes"]
            }

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        os.path.join(mesh_dir,"alpha_summary_new.csv"),
        index=False
    )


    # -----------------------------------------------------
    # Plot aggregated convergence curves
    # -----------------------------------------------------

    grouped = defaultdict(list)

    for fname,res in results.items():

        if len(res["alphas"])==0:
            continue

        actual = summary_df.loc[
            summary_df.filename==fname,
            "actual_volume"
        ].values[0]

        ratios = np.array(res["volumes"]) / actual

        grouped[extract_suture_type(fname)].append(
            (np.array(res["alphas"]),ratios)
        )

    plot_dir=os.path.join(mesh_dir,"plots")
    os.makedirs(plot_dir,exist_ok=True)

    for suture,curves in grouped.items():

        common = np.logspace(-4,4,200)

        interp=[]

        for a,r in curves:

            mask=np.isfinite(r)

            if np.sum(mask)<3:
                continue

            interp.append(
                np.interp(
                    np.log10(common),
                    np.log10(a[mask]),
                    r[mask]
                )
            )

        if len(interp)==0:
            continue

        mean=np.mean(interp,axis=0)

        plt.figure(figsize=(8,6))
        plt.plot(common,mean)
        plt.axhline(1,linestyle="--")
        plt.xscale("log")

        plt.xlabel("Alpha")
        plt.ylabel("Estimated / Actual Volume")
        plt.title(suture)

        plt.savefig(os.path.join(plot_dir,f"{suture}_convergence.png"))
        plt.close()


    with open(os.path.join(mesh_dir,"alpha_results_new.json"),"w") as f:
        json.dump(results,f,indent=2)

    print("Pipeline complete")

    return summary_df


# -----------------------------------------------------
# Run
# -----------------------------------------------------

if __name__ == "__main__":

    summary = run_pipeline(
        "D:/Marco_backups/pipeline/thrice_rescaled/",
        "D:/Marco_backups/pipeline/thrice_rescaled/mesh_rescaled_volume_centroid_2.csv",
        n_jobs=40
    )

    print(summary.head())
