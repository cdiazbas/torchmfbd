"""Run a torchmfbd configuration over scenes with a known truth, or compare two runs.

  python benchmarks/run.py [options] -o run.json
  python benchmarks/run.py --compare a.json b.json [--floor a_seed1.json]

A run fits every scene of the chosen sets with the yaml (default examples/spot_8542/kl.yaml) and the patch geometry of
the options, and writes one row per scene: object error, its error per frequency band, the per-frame PSF error, the
transfer error R, and the error of the object made from the true PSFs (the control). It stops if that control fails
(per scene, the object from the true PSFs may not be worse than the code's by more than 5 %; over each set, it must be
better), unless --no-control.

A comparison pairs the two runs scene by scene and prints, per set and overall: scenes where b is better, the mean
paired change of the object error, the sign test p, and the band errors. With --floor (the run of a with another --seed),
it also prints the noise floor: the mean |paired change| between two runs of the same configuration. A difference
inside the floor is not a difference. Runs from different devices are not compared: LBFGS amplifies the last bits of
the arithmetic, and two GPU types differ by 3-4 % (Mac and Linux CPUs by up to 0.24 in the per-scene PSF error).
"""
import argparse, copy, json, logging, math, os, platform, sys, time, pathlib, warnings
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M
import scenes as S


def device_name(gpu):
    import torch
    if gpu >= 0 and torch.cuda.is_available():
        return torch.cuda.get_device_name(gpu)
    return f"cpu {platform.machine()} {platform.processor() or platform.system()}"


def fit_scene(sc, base, a):
    import torch, torchmfbd
    conf = copy.deepcopy(base)
    conf["images"]["n_pixel"] = a.patch
    if a.border is not None:
        conf["images"]["apodization_border"] = a.border
    if a.loss:
        conf["optimization"]["loss_type"] = a.loss
    conf["optimization"]["gpu"] = a.gpu
    if a.coupling:                                # wavefronts of neighbouring patches coupled (regularization coupling)
        conf["regularization"] = dict(conf.get("regularization") or {}, coupling={"lambda": a.coupling, "n_modes": a.coupling_modes})
    if a.filter:
        conf["object1"]["image_filter"] = a.filter
    if a.nullspace:                               # experimental wavefront restriction (joint loss only)
        conf["psf"]["nullspace_d_r0"] = a.nullspace
    if a.seed:                                    # the wavefront starts at zero: another seed only matters with a
        conf.setdefault("initialization", {})["modes_std"] = 1e-3   # random start (1e-3 rad rms), as for the floor
    P4 = torchmfbd.Patchify4D()
    frames = sc["frames"]
    fr = torch.tensor((frames / frames.mean(axis=(-1, -2), keepdims=True))[None].astype("float32"))
    pt = P4.patchify(fr, patch_size=a.patch, stride_size=a.stride, flatten_sequences=True)
    torch.manual_seed(a.seed)
    t0 = time.time()
    d = torchmfbd.Deconvolution(conf); d.logger.setLevel(logging.ERROR)
    d.add_frames(pt, id_object=0, id_diversity=0, diversity=0.0)
    if a.su_rule:                                 # joint loss: object prior level from the noise, 250 (sigma/0.02)(12/N)
        sig = float(torch.mean(d.sigma[0])); nfr = d.frames[0].shape[1]
        d.config["object1"]["s_u_joint"] = 250.0 * (sig / 0.02) * max(12.0 / nfr, 1.0)
    d.set_neighbours(P4.neighbours())             # used only if the configuration couples neighbouring patches
    d.deconvolve(infer_object=False, optimizer=a.optimizer, simultaneous_sequences=10000, n_iterations=a.iterations)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_fit = time.time() - t0
    full = lambda o: np.squeeze(P4.unpatchify(o.cpu().float(), apodization=6, weight_type="cosine", weight_params=30).numpy())
    # the truth through the same patches, diffraction mask and recombination as the object
    tpp = P4.patchify(torch.tensor(sc["truth"][None, None].astype("float32")), patch_size=a.patch, stride_size=a.stride, flatten_sequences=True)[:, 0]
    tr = full(torch.fft.ifft2(torch.fft.fft2(tpp) * d.mask_diffraction_th[0].cpu()).real)
    o = full(d.obj[0])
    pe = torch.cat([b[0] for b in d.psf_seq], 0).cpu().numpy()
    # true PSFs of every patch and frame, and their OTFs with the frames' shifts on the patch grid
    n = frames.shape[-1]; J = frames.shape[0]
    if "psf_at" in sc:
        cy = P4.patchify(torch.tensor(np.broadcast_to(np.arange(n, dtype="float32")[:, None], (n, n)).copy())[None, None], patch_size=a.patch, stride_size=a.stride, flatten_sequences=True)[:, 0]
        cx = P4.patchify(torch.tensor(np.broadcast_to(np.arange(n, dtype="float32")[None, :], (n, n)).copy())[None, None], patch_size=a.patch, stride_size=a.stride, flatten_sequences=True)[:, 0]
        true_psfs = np.stack([np.stack([sc["psf_at"](j, float(cy[p].mean()), float(cx[p].mean())) for j in range(J)]) for p in range(cy.shape[0])])
    else:
        true_psfs = sc["psfs"]
    shifted = lambda q, sh: np.roll(M.fold_psf(q, a.patch), sh, (0, 1))
    if true_psfs.ndim == 3:                                                      # (frames, m, m)
        tp = np.stack([shifted(q, sh) for q, sh in zip(true_psfs, sc["shifts"])])
    else:                                                                        # (patches, frames, m, m)
        tp = np.stack([np.stack([shifted(q, sh) for q, sh in zip(qp, sc["shifts"])]) for qp in true_psfs])
    row = dict(scene=sc["name"], time_fit=t_fit, n_patches=int(pt.shape[0]),
               err_obj=M.object_error(o, tr), err_obj_bestscale=M.object_error_bestscale(o, tr), band_err=M.band_error(o, tr),
               err_psf=M.psf_error(pe, true_psfs), err_transfer=M.transfer_error(pe, np.fft.fft2(tp), tpp.numpy()))
    # control: the object from the true OTFs, with the run's own noise and (marginal loss) spectrum parameters
    otf = torch.fft.fft2(torch.tensor(tp, dtype=torch.float32), norm="ortho").to(d.device)
    if otf.ndim == 3:
        otf = otf[None].repeat(d.frames_ft[0].shape[0], 1, 1, 1)
    kw = {}
    if d.loss_type == "marginal":
        assert len(d.pars_raw_seq) == 1, "one batch of patches expected"
        kw = dict(pars_s0=d.pars_raw_seq[0][0], pars_s2=d.pars_raw_seq[0][1])
    with torch.no_grad():
        _, _, ot = d.compute_object([d.frames_ft[0].to(d.device)], [otf], [d.sigma[0].to(d.device)], [d.plane[0].to(d.device)], **kw)
    row["err_obj_true_psf"] = M.object_error(full(ot[0]), tr)
    if not a.no_control and row["err_obj_true_psf"] > M.CONTROL_SCENE_TOL * row["err_obj"]:
        raise RuntimeError(f"control failed {sc['name']}: the object from the true PSFs ({row['err_obj_true_psf']:.4f}) is more "
                           f"than 5 % worse than the code's ({row['err_obj']:.4f}); check the truth units and the PSF reference")
    return row


def fit_time_scene(sc, base, a):
    """A scene with time steps (set aniso-time): the patches of every step fitted together with filter_loss
    wiener_time and the time term of weight --time-lambda; metrics per step, then averaged over the steps."""
    import torch, torchmfbd
    conf = copy.deepcopy(base)
    conf["images"]["n_pixel"] = a.patch
    if a.border is not None:
        conf["images"]["apodization_border"] = a.border
    if a.loss:
        conf["optimization"]["loss_type"] = a.loss
    conf["optimization"]["gpu"] = a.gpu
    conf["optimization"]["filter_loss"] = "wiener_time"
    conf["regularization"] = dict(conf.get("regularization") or {}, time={"lambda": a.time_lambda})
    if a.coupling:
        conf["regularization"]["coupling"] = {"lambda": a.coupling, "n_modes": a.coupling_modes}
    elif "coupling" in conf["regularization"] and a.no_yaml_coupling:
        conf["regularization"].pop("coupling")
    if a.filter:
        conf["object1"]["image_filter"] = a.filter
    if a.seed:
        conf.setdefault("initialization", {})["modes_std"] = 1e-3
    P4 = torchmfbd.Patchify4D()
    frames = sc["frames"]                                                # (T, J, n, n)
    T, J, n = frames.shape[0], frames.shape[1], frames.shape[-1]
    fr = frames / frames.mean(axis=(-1, -2), keepdims=True)
    pt = torch.stack([P4.patchify(torch.tensor(fr[t][None].astype("float32")), patch_size=a.patch, stride_size=a.stride,
                                  flatten_sequences=True) for t in range(T)], dim=1)          # (L, T, J, m, m)
    pairs = P4.neighbours()
    L = pt.shape[0]
    torch.manual_seed(a.seed)
    t0 = time.time()
    d = torchmfbd.Deconvolution(conf); d.logger.setLevel(logging.ERROR)
    d.add_frames(pt, id_object=0, id_diversity=0, diversity=0.0)
    if a.su_rule:
        sig = float(torch.mean(d.sigma[0])); nfr = d.frames[0].shape[1]
        d.config["object1"]["s_u_joint"] = 250.0 * (sig / 0.02) * max(12.0 / nfr, 1.0)
    d.set_neighbours(pairs)
    d.deconvolve(infer_object=False, optimizer=a.optimizer, simultaneous_sequences=10000, n_iterations=a.iterations)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_fit = time.time() - t0
    full = lambda o: np.squeeze(P4.unpatchify(o.cpu().float(), apodization=6, weight_type="cosine", weight_params=30).numpy())
    obj = d.obj[0].reshape(L, T, a.patch, a.patch)
    pe = torch.cat([b[0] for b in d.psf_seq], 0).cpu().numpy().reshape(L, T, J, a.patch, a.patch)
    cy = P4.patchify(torch.tensor(np.broadcast_to(np.arange(n, dtype="float32")[:, None], (n, n)).copy())[None, None], patch_size=a.patch, stride_size=a.stride, flatten_sequences=True)[:, 0]
    cx = P4.patchify(torch.tensor(np.broadcast_to(np.arange(n, dtype="float32")[None, :], (n, n)).copy())[None, None], patch_size=a.patch, stride_size=a.stride, flatten_sequences=True)[:, 0]
    centres = [(float(cy[p].mean()), float(cx[p].mean())) for p in range(L)]
    per_step, tps, trs = [], [], []
    for t in range(T):
        tpp = P4.patchify(torch.tensor(sc["truth"][t][None, None].astype("float32")), patch_size=a.patch, stride_size=a.stride, flatten_sequences=True)[:, 0]
        tr = full(torch.fft.ifft2(torch.fft.fft2(tpp) * d.mask_diffraction_th[0].cpu()).real); trs.append(tr)
        o = full(obj[:, t])
        true_psfs = np.stack([np.stack([sc["psf_at"][t](j, y, x) for j in range(J)]) for (y, x) in centres])
        tp = np.stack([np.stack([np.roll(M.fold_psf(q, a.patch), sh, (0, 1)) for q, sh in zip(qp, sc["shifts"][t])]) for qp in true_psfs])
        tps.append(tp)
        per_step.append(dict(err_obj=M.object_error(o, tr), band_err=M.band_error(o, tr), err_psf=M.psf_error(pe[:, t], true_psfs),
                             err_transfer=M.transfer_error(pe[:, t], np.fft.fft2(tp), tpp.numpy())))
    row = dict(scene=sc["name"], time_fit=t_fit, n_patches=int(L), n_steps=T, err_obj_steps=[r["err_obj"] for r in per_step],
               err_obj=float(np.mean([r["err_obj"] for r in per_step])), band_err=list(np.mean([r["band_err"] for r in per_step], axis=0)),
               err_psf=float(np.mean([r["err_psf"] for r in per_step])), err_transfer=float(np.mean([r["err_transfer"] for r in per_step])))
    # control: the object from the true OTFs of every patch and step, (s t) order as the library flattens them
    otf = torch.fft.fft2(torch.tensor(np.stack(tps, axis=1).reshape(L * T, J, a.patch, a.patch), dtype=torch.float32), norm="ortho").to(d.device)
    with torch.no_grad():
        _, _, ot = d.compute_object([d.frames_ft[0].to(d.device)], [otf], [d.sigma[0].to(d.device)], [d.plane[0].to(d.device)])
    ot = ot[0].reshape(L, T, a.patch, a.patch)
    row["err_obj_true_psf"] = float(np.mean([M.object_error(full(ot[:, t]), trs[t]) for t in range(T)]))
    if not a.no_control and row["err_obj_true_psf"] > M.CONTROL_SCENE_TOL * row["err_obj"]:
        raise RuntimeError(f"control failed {sc['name']}: the object from the true PSFs ({row['err_obj_true_psf']:.4f}) is more "
                           f"than 5 % worse than the code's ({row['err_obj']:.4f})")
    return row


def run(a):
    import yaml
    with open(a.config) as f:
        base = yaml.safe_load(f)
    base.pop("object2", None)
    rows = []
    for s in a.sets.split(","):
        for sc in S.scene_set(s, n_frames=a.frames, noise=a.noise, alt=a.alt, h5_path=a.h5, holdout=a.holdout, seq_path=a.seq):
            if a.only and not any(sc["name"].startswith(o) for o in a.only.split(",")):
                continue
            r = (fit_time_scene if sc.get("time") else fit_scene)(sc, base, a); r["set"] = s; rows.append(r)
            print(f"{r['scene']:22s} object {r['err_obj']:.4f} (true PSFs {r['err_obj_true_psf']:.4f})  PSF {r['err_psf']:.3f}"
                  f"  R {r['err_transfer']:.3f}  {r['time_fit']:.1f} s", flush=True)
        R = [r for r in rows if r["set"] == s]
        if not a.no_control:          # over the set, the true PSFs must give a clearly better object (a units error does not)
            M.check_control(np.mean([r["err_obj_true_psf"] for r in R]), np.mean([r["err_obj"] for r in R]), f"set {s}")
    info = dict(device=device_name(a.gpu), args=vars(a), date=time.strftime("%Y-%m-%d %H:%M"))
    import torchmfbd
    info["torchmfbd"] = str(pathlib.Path(torchmfbd.__file__).resolve().parent)
    json.dump(dict(info=info, rows=rows), open(a.output, "w"), indent=1)
    print(f"wrote {a.output}: {len(rows)} scenes, mean object error {np.mean([r['err_obj'] for r in rows]):.4f}")


def sign_p(wins, n):
    """Two-sided exact sign test."""
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def compare(a_path, b_path, floor_path=None):
    A, B = json.load(open(a_path)), json.load(open(b_path))
    if A["info"]["device"] != B["info"]["device"]:
        sys.exit(f"not comparable: {A['info']['device']} against {B['info']['device']} (run both on the same device)")
    ra = {r["scene"]: r for r in A["rows"]}; rb = {r["scene"]: r for r in B["rows"]}
    common = [s for s in ra if s in rb]
    floor = None
    if floor_path:
        F = json.load(open(floor_path)); assert F["info"]["device"] == A["info"]["device"], "floor run on another device"
        rf = {r["scene"]: r for r in F["rows"]}
        floor = float(np.mean([abs(100 * (rf[s]["err_obj"] / ra[s]["err_obj"] - 1)) for s in common if s in rf]))
    groups = {}
    for s in common:
        groups.setdefault(ra[s].get("set", "all"), []).append(s)
    groups["all"] = common
    print(f"a = {a_path}\nb = {b_path}\ndevice {A['info']['device']}" + (f"; noise floor +-{floor:.1f} % (|a - a, other seed|)" if floor is not None else ""))
    for g, sc in groups.items():
        ch = np.array([100 * (rb[s]["err_obj"] / ra[s]["err_obj"] - 1) for s in sc])
        w, l = int((ch < 0).sum()), int((ch > 0).sum())             # ties (identical fits) carry no sign
        ba = np.mean([ra[s]["band_err"] for s in sc], axis=0); bb = np.mean([rb[s]["band_err"] for s in sc], axis=0)
        verdict = "" if floor is None else ("  beyond the noise floor" if abs(ch.mean()) > floor else "  within the noise floor")
        print(f"{g:12s} n={len(sc):3d}  object {np.mean([ra[s]['err_obj'] for s in sc]):.4f} -> {np.mean([rb[s]['err_obj'] for s in sc]):.4f}"
              f"  change {ch.mean():+6.1f} %  b better in {w}/{len(sc)} (ties {len(sc) - w - l})  sign p={sign_p(w, w + l) if w + l else 1.0:.2g}"
              f"  R {np.mean([ra[s]['err_transfer'] for s in sc]):.3f} -> {np.mean([rb[s]['err_transfer'] for s in sc]):.3f}"
              f"  >25 % worse: {int((ch > 25).sum())}{verdict}")
        print(f"{'':12s} band error " + "  ".join(f"{lo:.2f}-{hi:.2f}: {x:.2f}->{y:.2f}" for (lo, hi), x, y in zip(M.BANDS, ba, bb)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("--floor", help="with --compare: the run of A with another --seed")
    p.add_argument("--config", default=str(HERE.parent / "examples" / "spot_8542" / "kl.yaml"))
    p.add_argument("--sets", default="synth,aniso-synth", help="comma list of synth, aniso-synth, real, aniso-real, aniso-time, real-time")
    p.add_argument("--patch", type=int, default=128); p.add_argument("--stride", type=int, default=32)
    p.add_argument("--border", type=int, help="apodization border (default: the yaml's)")
    p.add_argument("--loss", choices=["joint", "marginal"], help="default: the yaml's")
    p.add_argument("--optimizer", default="lbfgs"); p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--coupling", type=float, default=0.0, help="lambda of the coupling of neighbouring patches (0: off)")
    p.add_argument("--coupling-modes", type=int, default=10)
    p.add_argument("--filter", choices=["scharmer", "garrote", "tophat"], help="object filter (default: the yaml's)")
    p.add_argument("--su-rule", action="store_true", help="joint loss: s_u_joint = 250 (sigma/0.02)(12/N) from the data")
    p.add_argument("--nullspace", type=float, default=0.0, help="D/r0 of the experimental wavefront restriction (0: off)")
    p.add_argument("--time-lambda", type=float, default=1.5e-4, help="time sets: weight of the time term (regularization time)")
    p.add_argument("--no-yaml-coupling", action="store_true", help="drop a coupling entry of the yaml (to measure without it)")
    p.add_argument("--frames", type=int, default=12); p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--alt", type=float, default=5000.0, help="height of the upper seeing layer of the aniso sets, m")
    p.add_argument("--seed", type=int, default=0, help="seed of the initial wavefront (another seed gives the noise floor)")
    p.add_argument("--gpu", type=int, default=0, help="-1 for the CPU")
    p.add_argument("--h5", help="the burst file, for the real sets")
    p.add_argument("--seq", help="the time sequence file, for the real-time set")
    p.add_argument("--no-control", action="store_true")
    p.add_argument("--holdout", action="store_true", help="the hold-out scenes (other seeds): to confirm a choice")
    p.add_argument("--only", help="comma list of scene-name prefixes to keep (a quick screen)")
    p.add_argument("-o", "--output", default="run.json")
    a = p.parse_args()
    warnings.simplefilter("ignore"); os.environ.setdefault("TQDM_DISABLE", "1")
    if a.compare:
        compare(*a.compare, a.floor)
    else:
        run(a)


if __name__ == "__main__":
    main()
