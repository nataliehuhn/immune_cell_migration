"""
This script takes folders with clickpoints databases within them. Every clickpoints database requires a mask segmenting
the cells in every frame. This script creates tracks for the specified objects and saves them to the clickpoints
database. Those tracks are based on the overlap from an object in one frame to the next frame and are subjected to
stitching with the provided parameters.
"""
def run_tracking(cdb_path, cdb, frames, start_frame=1):
    """
    CPU-only SAM2 hybrid tracker:
      - Input: ClickPoints cdb (images + masks)
      - Seeding: prefer cdb mask (ANY nonzero label) on your image layer first; robust fallback across layers;
                 if nothing found, threshold fallback (won't crash)
      - Tracking: SAM2 propagation (forced CPU)
      - Z: NaN for now
      - Drift: OFF by default (matches pipeline run with drift_corr=False). Enable below if you have drift files.
      - Output: tracks written back to the same .cdb (markers on "MinProj")
    """
    # ------------ force CPU & imports ------------
    import os, gc, shutil, warnings
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU for the process
    warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")  # quiet SAM2 ext warn

    import numpy as np
    from pathlib import Path
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects
    from skimage.filters import threshold_yen
    from skimage.io import imread
    import imageio.v2 as imageio

    # robust SAM2 import (supports both layouts)
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except Exception:
        try:
            from sam2.video_predictor import SAM2VideoPredictor  # older layout
        except Exception as e:
            raise RuntimeError(
                "Could not import SAM2VideoPredictor. Install SAM2 in this env:\n"
                '  python -m pip install --no-cache-dir "git+https://github.com/facebookresearch/segment-anything-2.git"'
            ) from e

    # ------------ Config (CPU-friendly) ------------
    MODEL_ID       = "facebook/sam2.1-hiera-large"
    BATCH_SIZE     = 30
    OVERLAP        = 8
    MIN_OBJ_AREA   = 30
    SEEDING_MODE   = "cdb_mask_all"   # try cdb mask first; fallback to threshold automatically if missing
    TRACK_NAME     = "nk_tracks_sam2"
    TRACK_COLOR    = "#00ffff"
    REMOVE_SHORT   = True
    MIN_TRACK_LEN  = 4

    USE_DRIFT      = False            # <- keep False to match drift_corr=False. Set True to enable drift logic.
    DRIFT_DIR      = os.environ.get("DRIFT_DIR", r"E:\LucasScricpts\calculatedDrifts")

    # *** IMPORTANT: your data is on this image layer ***
    IMAGE_LAYER    = 1  # export frames and look up masks on this layer first

    # Prefer masks on this numeric layer (often the same as IMAGE_LAYER), but we search broadly:
    PREFERRED_MASK_LAYER   = 1
    EXTRA_LAYER_CANDIDATES = tuple(range(0, 10))  # probe numeric layers 0..9

    # Optional: downscale frames to speed up CPU. Coords are mapped back to original size.
    DOWNSCALE      = 1.0  # e.g. 0.5 to halve width/height

    # ------------ Paths / output ------------
    def _find_cdb_file():
        for attr in ("filename", "file", "path", "name"):
            v = getattr(cdb, attr, None)
            if isinstance(v, str) and v.lower().endswith(".cdb"):
                return v
        p = Path(str(cdb_path))
        if p.is_file() and str(p).lower().endswith(".cdb"):
            return str(p)
        if p.is_dir():
            cand = list(p.glob("*-*_pos*.cdb")) or list(p.glob("*.cdb"))
            if cand:
                return str(cand[0])
        return None

    cdb_file = _find_cdb_file() or "database.cdb"
    out_root = Path(cdb_file).with_suffix("").as_posix()
    output_dir = Path(out_root + "_sam2_outputs_cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "last_processed_frame.txt"
    resume_file     = output_dir / "resume_prompts.npz"

    # ------------ Helpers ------------
    def _to_uint8(img):
        if img.dtype == np.uint8:
            return img
        a = img.astype(np.float32)
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx <= mn:
            return np.zeros_like(a, dtype=np.uint8)
        a = (a - mn) / (mx - mn)
        return np.clip(a * 255.0, 0, 255).astype(np.uint8)

    def _load_drift(image_shape, total_frames):
        if not USE_DRIFT:
            return None, None, image_shape
        drift_file = None
        cand = Path(DRIFT_DIR) / (Path(cdb_file).stem + ".txt")
        if cand.exists():
            drift_file = cand
        if drift_file is None:
            txts = list(Path(DRIFT_DIR).glob("*.txt"))
            if txts:
                drift_file = txts[0]
        if drift_file is None or not drift_file.exists():
            return None, None, image_shape

        drift_arr = np.loadtxt(str(drift_file), delimiter=",")  # rows [dy, dx]
        if drift_arr.shape[0] < total_frames:
            pad = np.zeros((total_frames - drift_arr.shape[0], 2), dtype=drift_arr.dtype)
            drift_arr = np.vstack([drift_arr, pad])
        elif drift_arr.shape[0] > total_frames:
            drift_arr = drift_arr[:total_frames]

        drift_sum = np.cumsum(drift_arr, axis=0)
        drift_sum = np.insert(drift_sum, 0, np.array([0, 0]), axis=0)  # 1-based alignment

        max_x_dr = np.max(drift_sum[:, 1]); min_x_dr = np.min(drift_sum[:, 1])
        max_y_dr = np.max(drift_sum[:, 0]); min_y_dr = np.min(drift_sum[:, 0])

        H0, W0 = image_shape
        drift_h = int(H0 - max_y_dr + min_y_dr)
        drift_w = int(W0 - max_x_dr + min_x_dr)
        drift_values = [max_y_dr, max_x_dr, drift_h, drift_w]  # y0, x0, H, W
        return drift_sum, drift_values, (drift_h, drift_w)

    def _crop_with_drift(img, frame_1b, drift_sum, drift_values):
        if drift_sum is None or drift_values is None:
            return img
        y0, x0, H, W = drift_values
        dy, dx = drift_sum[frame_1b]
        ys, xs = int(y0 - dy), int(x0 - dx)
        return img[ys:ys + int(H), xs:xs + int(W)]

    def _maybe_downscale(img_u8):
        if DOWNSCALE == 1.0:
            return img_u8
        try:
            import cv2
            h, w = img_u8.shape[:2]
            nh, nw = int(round(h * DOWNSCALE)), int(round(w * DOWNSCALE))
            return cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            s = max(1, int(round(1.0 / DOWNSCALE)))
            return img_u8[::s, ::s]

    def _export_batch(temp_dir, batch_start, batch_end, drift_sum, drift_values):
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for local_idx, gidx in enumerate(range(batch_start, batch_end)):
            im = cdb.getImage(frame=gidx, layer=IMAGE_LAYER).data  # FIX: explicit image layer
            if im.ndim == 3:
                im = im[..., 0]
            im8 = _to_uint8(im)
            frame_1b = gidx + 1
            im8c = _crop_with_drift(im8, frame_1b, drift_sum, drift_values)
            im8c = _maybe_downscale(im8c)
            imageio.imwrite(str(temp_dir / f"{local_idx:06d}.jpg"), im8c)

    def _centroid_area(binary_mask_uint8):
        lab = label(binary_mask_uint8, connectivity=1)
        if lab.max() == 0:
            return None
        rp = regionprops(lab)
        reg = max(rp, key=lambda r: r.area)
        (y, x) = reg.centroid
        return float(y), float(x), float(reg.area)

    def _append_pos(tracks, oid, frame_1b, y, x, z, area):
        if oid not in tracks:
            tracks[oid] = {"start_frame": frame_1b, "pos": []}  # start_frame is 1-based
        start = tracks[oid]["start_frame"]
        while len(tracks[oid]["pos"]) < (frame_1b - start):
            tracks[oid]["pos"].append([np.nan, np.nan, np.nan, np.nan])
        tracks[oid]["pos"].append([y, x, z, area])

    def _remove_short_tracks(tracks, min_len=4):
        drop = []
        for tid, v in tracks.items():
            pos = np.asarray(v["pos"], dtype=float)
            if pos.ndim != 2 or pos.size == 0:
                drop.append(tid); continue
            if np.sum(~np.isnan(pos[:, 0])) < min_len:
                drop.append(tid)
        for t in drop:
            tracks.pop(t, None)

    def _add_drift_and_upsample_back(tracks, drift_sum, drift_values):
        """Map cropped+downscaled coords back to ORIGINAL image coords."""
        scale = (1.0 / DOWNSCALE)
        if drift_sum is None or drift_values is None:
            if scale != 1.0:
                out = {}
                for tid, v in tracks.items():
                    newp = []
                    for p in v["pos"]:
                        if np.any(np.isnan(p)):
                            newp.append([np.nan, np.nan, np.nan, np.nan])
                        else:
                            newp.append([p[0] * scale, p[1] * scale, p[2], p[3]])
                    out[tid] = {"start_frame": v["start_frame"], "pos": newp}
                return out
            return tracks

        y0, x0, H, W = drift_values
        out = {}
        for tid, v in tracks.items():
            f1 = v["start_frame"]
            newp = []
            for p in v["pos"]:
                if np.any(np.isnan(p)):
                    newp.append([np.nan, np.nan, np.nan, np.nan])
                else:
                    dy, dx = drift_sum[f1]
                    yg_cropped = p[0] * scale
                    xg_cropped = p[1] * scale
                    yg = (y0 - dy) + yg_cropped
                    xg = (x0 - dx) + xg_cropped
                    newp.append([float(yg), float(xg), float(p[2]), float(p[3])])
                f1 += 1
            out[tid] = {"start_frame": v["start_frame"], "pos": newp}
        return out

    def _write_tracks_to_cdb(cdb_obj, track_name, track_color, tracks):
        # NOTE: ClickPoints frames are 0-based; our 'start_frame' is 1-based -> subtract 1 here
        try:
            cdb_obj.deleteMarkerTypes(name=track_name)
            cdb_obj.setMarkerType(name=track_name, color=track_color, mode=4)
        except Exception:
            pass
        for _, v in tracks.items():
            new_tr = cdb_obj.setTrack(type=track_name)
            start = v["start_frame"]  # 1-based
            pos = np.asarray(v["pos"], dtype=float)
            if pos.ndim != 2 or pos.size == 0:
                continue
            valid = np.nonzero(~np.isnan(pos[:, 0]))[0]
            if valid.size == 0:
                continue
            frames_out = (start - 1 + valid).tolist()  # FIX: write 0-based frames
            ys = pos[valid, 0].tolist()
            xs = pos[valid, 1].tolist()
            cdb_obj.setMarkers(frame=frames_out, x=xs, y=ys, track=new_tr, type=track_name, layer="MinProj")

    def _get_mask_for_frame(cdb_obj, frame_idx_0b, drift_sum, drift_values, downscale):
        """
        Try layers in this priority:
          1) IMAGE_LAYER (where we export frames)
          2) PREFERRED_MASK_LAYER (often same as IMAGE_LAYER)
          3) common named layers ("MaxIndices","MinIndices","MinProj")
          4) a broad set of numeric layers (0..9)
          5) default (None)
        Return (mask_uint8, H, W) or (None, None, None).
        """
        cand = []
        def _add(x):
            if x not in cand:
                cand.append(x)

        _add(IMAGE_LAYER)
        _add(PREFERRED_MASK_LAYER)
        for name in ("MaxIndices", "MinIndices", "MinProj"):
            _add(name)
        for lay in EXTRA_LAYER_CANDIDATES:
            _add(lay)
        _add(None)

        for layer in cand:
            try:
                img = (cdb_obj.getImage(frame=frame_idx_0b, layer=layer)
                       if layer is not None else cdb_obj.getImage(frame=frame_idx_0b))
                m = cdb_obj.getMask(image=img)
                if m is None:
                    continue
                mask_map = m.data  # int labels
                frame_1b = frame_idx_0b + 1
                mask_map = _crop_with_drift(mask_map, frame_1b, drift_sum, drift_values)
                if downscale != 1.0:
                    try:
                        import cv2
                        h, w = mask_map.shape
                        mask_map = cv2.resize(mask_map.astype(np.uint8),
                                              (int(round(w * downscale)), int(round(h * downscale))),
                                              interpolation=cv2.INTER_NEAREST)
                    except Exception:
                        s = max(1, int(round(1.0 / downscale)))
                        mask_map = mask_map[::s, ::s]
                print(f"[INFO] Using cdb mask from layer={layer!r} for frame {frame_idx_0b}.")
                return mask_map.astype(np.uint8), mask_map.shape[0], mask_map.shape[1]
            except Exception:
                continue

        return None, None, None

    # ------------ Prep & predictor ------------
    total_frames = int(frames)
    im0 = cdb.getImage(frame=0, layer=IMAGE_LAYER).data  # FIX: explicit layer
    if im0.ndim == 3:
        im0 = im0[..., 0]
    im0_u8 = _to_uint8(im0)

    drift_sum, drift_values, cropped_shape = _load_drift(im0_u8.shape, total_frames)
    if DOWNSCALE != 1.0:
        h, w = cropped_shape
        cropped_shape = (int(round(h * DOWNSCALE)), int(round(w * DOWNSCALE)))

    predictor = SAM2VideoPredictor.from_pretrained(MODEL_ID, device="cpu")  # FORCE CPU

    last_processed = -1
    if checkpoint_file.exists():
        try:
            last_processed = int(checkpoint_file.read_text().strip())
        except Exception:
            last_processed = -1

    obj_id = 1
    tracks = {}

    # ------------ Batching loop ------------
    for batch_start in range(start_frame - 1, total_frames, BATCH_SIZE - OVERLAP):
        batch_end = min(batch_start + BATCH_SIZE, total_frames)
        if batch_end - 1 <= last_processed:
            print(f"⏭️ Skipping batch {batch_start}-{batch_end-1}")
            continue

        print(f"🚀 Processing batch {batch_start}-{batch_end-1}...")
        temp_dir = output_dir / "temp_batch_frames"
        _export_batch(temp_dir, batch_start, batch_end, drift_sum, drift_values)

        state = predictor.init_state(video_path=str(temp_dir), offload_video_to_cpu=True)
        predictor.reset_state(state)

        # Seed first processed batch
        if batch_start == (start_frame - 1):
            if SEEDING_MODE == "cdb_mask_all":
                mask_map, _, _ = _get_mask_for_frame(cdb, batch_start, drift_sum, drift_values, DOWNSCALE)
                if mask_map is None:
                    print("[WARN] No mask found on common layers. Falling back to threshold seeding.")
                    # --- fallback to threshold ---
                    f0 = imread(str(temp_dir / "000000.jpg"))
                    if f0.ndim == 3:
                        f0 = f0.mean(axis=2)
                    f0n = f0 / 255.0 if f0.max() > 1.5 else f0
                    thr = threshold_yen(f0n)
                    mask = remove_small_objects(f0n < thr, min_size=MIN_OBJ_AREA)
                    lab  = label(mask, connectivity=1)
                    regs = regionprops(lab)
                    areas = [r.area for r in regs] or [1]
                    a_mean = float(np.mean(areas))
                    for r in regs:
                        if r.area > 2.0 * a_mean:
                            continue
                        binary = (lab == r.label).astype(np.uint8)
                        predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=int(obj_id), mask=binary)
                        obj_id += 1
                else:
                    tgt = remove_small_objects((mask_map != 0), min_size=MIN_OBJ_AREA)
                    lab = label(tgt, connectivity=1)
                    for comp in range(1, lab.max() + 1):
                        binary = (lab == comp).astype(np.uint8)
                        predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=int(obj_id), mask=binary)
                        obj_id += 1

            elif SEEDING_MODE == "threshold":
                f0 = imread(str(temp_dir / "000000.jpg"))
                if f0.ndim == 3:
                    f0 = f0.mean(axis=2)
                f0n = f0 / 255.0 if f0.max() > 1.5 else f0
                thr = threshold_yen(f0n)
                mask = remove_small_objects(f0n < thr, min_size=MIN_OBJ_AREA)
                lab  = label(mask, connectivity=1)
                regs = regionprops(lab)
                areas = [r.area for r in regs] or [1]
                a_mean = float(np.mean(areas))
                for r in regs:
                    if r.area > 2.0 * a_mean:
                        continue
                    binary = (lab == r.label).astype(np.uint8)
                    predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=int(obj_id), mask=binary)
                    obj_id += 1

            else:
                raise ValueError("SEEDING_MODE must be 'cdb_mask_all' or 'threshold'")

        else:
            # Resume from previous batch snapshot
            if resume_file.exists():
                dat = np.load(resume_file, allow_pickle=True)
                mask_resume = dat["mask"]
                obj_ids_resume = dat["obj_ids"]
                for oid in np.unique(obj_ids_resume):
                    if int(oid) == 0:
                        continue
                    binary = (mask_resume == int(oid)).astype(np.uint8)
                    predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=int(oid), mask=binary)
            else:
                raise RuntimeError("❌ No resume file found — can't continue tracking.")

        # Propagate on CPU
        local_len = batch_end - batch_start
        resume_save_idx = max(0, local_len - OVERLAP - 1)

        mask_out_for_resume = None
        obj_ids_for_resume  = None

        for out_idx, out_ids, out_logits in predictor.propagate_in_video(state):
            true_idx = batch_start + out_idx
            if true_idx <= last_processed:
                continue

            # Collect tracks in cropped(+downscaled) coords
            h, w = cropped_shape
            mask_out = np.zeros((h, w), dtype=np.int32)
            frame_1b = true_idx + 1  # keep 1-based internally

            for oid, logit in zip(out_ids, out_logits):
                bm = (logit > 0.0).cpu().numpy()[0].astype(np.uint8)
                ca = _centroid_area(bm)
                if ca is not None:
                    y, x, area = ca
                    _append_pos(tracks, int(oid), frame_1b, y, x, np.nan, area)
                mask_out[bm.astype(bool)] = int(oid)

            checkpoint_file.write_text(str(true_idx))
            print(f"✅ Saved frame {true_idx}/{total_frames - 1}")

            if out_idx == resume_save_idx:
                mask_out_for_resume = mask_out
                obj_ids_for_resume  = np.array(out_ids)

        if mask_out_for_resume is not None:
            np.savez(str(resume_file), mask=mask_out_for_resume, obj_ids=obj_ids_for_resume)

        shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------ Finalize & write ------------
    if REMOVE_SHORT:
        _remove_short_tracks(tracks, min_len=MIN_TRACK_LEN)

    # Re-add drift + upsample back to original image coords (drift off -> identity)
    drift_sum, drift_values, _ = _load_drift(_to_uint8(cdb.getImage(frame=0, layer=IMAGE_LAYER).data).shape, total_frames)
    tracks_global = _add_drift_and_upsample_back(tracks, drift_sum, drift_values)

    # Write back to the SAME cdb object (frames fixed to 0-based)
    _write_tracks_to_cdb(cdb, TRACK_NAME, TRACK_COLOR, tracks_global)

    print("------done tracking (SAM2 CPU hybrid)------")
    return TRACK_NAME
