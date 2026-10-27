#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playermodel Maker  -  double-click to run
=========================================
Builds playermodels for the maze game (the <id>/meta.json + model.bin.gz + PNG
layout that the game loads from the playermodels repo).

  1. Add one or more .smd files (reference SMDs: body, head, hands ...).
     They are merged into ONE model; bones are matched by name.
  2. Every material found in the SMDs shows up in the table. Map each one to a
     .vtf (or .png/.jpg/.tga) or a flat colour.  "Auto-match folder" finds
     VTFs (and follows VMT $basetexture) by material name for you.
  3. Check the live preview (drag = rotate, right-drag = pan, wheel = zoom).
  4. Pick the output folder (your local clone of the playermodels repo) and
     press Export.  models.json is created/updated automatically.

First launch installs numpy + pillow automatically if they are missing.
"""
import os, sys, re, json, gzip, math, struct, subprocess, traceback


# --------------------------------------------------------------------------
#  dependency bootstrap (so a plain double-click just works)
# --------------------------------------------------------------------------
def _ensure_deps():
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("pillow")
    if not missing:
        return
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title("Playermodel Maker")
    tk.Label(root, padx=30, pady=25,
             text="First run: installing " + " + ".join(missing) +
                  " ...\n(this only happens once - please wait)").pack()
    root.update()
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000  # no console window
    ok, out = False, ""
    for extra in (["--user"], []):
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install"] + extra + missing,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, text=True, **kw)
            ok, out = r.returncode == 0, r.stdout
        except Exception as e:  # noqa: BLE001
            ok, out = False, str(e)
        if ok:
            break
    root.destroy()
    if not ok:
        r2 = tk.Tk()
        r2.withdraw()
        messagebox.showerror("Playermodel Maker",
                             "Could not install " + ", ".join(missing) +
                             ".\nRun:  pip install " + " ".join(missing) + "\n\n" + out[-1500:])
        sys.exit(1)
    import site, importlib
    try:
        sys.path.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    importlib.invalidate_caches()


_ensure_deps()

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser


# --------------------------------------------------------------------------
#  VTF reader  (Valve Texture Format -> PIL image)
# --------------------------------------------------------------------------
_BPP = {0: 4, 1: 4, 2: 3, 3: 3, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 3, 10: 3, 11: 4, 12: 4,
        16: 4, 17: 2, 18: 2, 19: 2, 21: 2, 22: 2, 23: 4, 24: 8, 25: 8, 26: 4}
_FMT_NAMES = {0: "RGBA8888", 1: "ABGR8888", 2: "RGB888", 3: "BGR888", 4: "RGB565", 5: "I8",
              6: "IA88", 7: "P8", 8: "A8", 11: "ARGB8888", 12: "BGRA8888", 13: "DXT1",
              14: "DXT3", 15: "DXT5", 16: "BGRX8888", 17: "BGR565", 18: "BGRX5551",
              19: "BGRA4444", 20: "DXT1_ONEBITALPHA", 21: "BGRA5551"}


def _img_size(w, h, fmt):
    if fmt in (13, 20):
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 8
    if fmt in (14, 15):
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16
    return w * h * _BPP.get(fmt, 4)


def _decode_dxt(data, w, h, fmt):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    bs = 8 if fmt in (13, 20) else 16
    n = bw * bh
    a = np.frombuffer(data, dtype=np.uint8, count=n * bs).reshape(n, bs).astype(np.uint32)
    cb = a[:, bs - 8:]
    c0 = cb[:, 0] | (cb[:, 1] << 8)
    c1 = cb[:, 2] | (cb[:, 3] << 8)

    def to_rgb(c):
        r = ((c >> 11) & 31).astype(np.int32)
        g = ((c >> 5) & 63).astype(np.int32)
        b = (c & 31).astype(np.int32)
        return np.stack([(r * 255 + 15) // 31, (g * 255 + 31) // 63, (b * 255 + 15) // 31], 1)

    col0, col1 = to_rgb(c0), to_rgb(c1)
    four = (c0 > c1)[:, None]
    pal = np.zeros((n, 4, 3), np.int32)
    pal[:, 0] = col0
    pal[:, 1] = col1
    pal[:, 2] = np.where(four, (2 * col0 + col1) // 3, (col0 + col1) // 2)
    pal[:, 3] = np.where(four, (col0 + 2 * col1) // 3, 0)
    bits = cb[:, 4] | (cb[:, 5] << 8) | (cb[:, 6] << 16) | (cb[:, 7] << 24)
    idx = ((bits[:, None] >> (2 * np.arange(16, dtype=np.uint32))[None, :]) & 3).astype(np.intp)
    rgb = pal[np.arange(n)[:, None], idx]  # (n,16,3)

    if fmt == 13:
        alpha = np.full((n, 16), 255, np.int32)
    elif fmt == 20:
        alpha = np.full((n, 16), 255, np.int32)
        alpha[(~four) & (idx == 3)] = 0
    elif fmt == 14:
        ab = a[:, 0:8]
        alpha = (np.stack([ab & 15, ab >> 4], 2).reshape(n, 16) * 17).astype(np.int32)
    else:  # DXT5
        a0 = a[:, 0].astype(np.int32)
        a1 = a[:, 1].astype(np.int32)
        b48 = np.zeros(n, np.uint64)
        for k in range(6):
            b48 |= a[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
        i3 = ((b48[:, None] >> (3 * np.arange(16, dtype=np.uint64))[None, :]) & np.uint64(7)).astype(np.intp)
        ap = np.zeros((n, 8), np.int32)
        ap[:, 0], ap[:, 1] = a0, a1
        gt = a0 > a1
        for i in range(2, 8):
            if i < 6:
                alt = ((6 - i) * a0 + (i - 1) * a1) // 5
            else:
                alt = np.zeros_like(a0) if i == 6 else np.full_like(a0, 255)
            ap[:, i] = np.where(gt, ((8 - i) * a0 + (i - 1) * a1) // 7, alt)
        alpha = ap[np.arange(n)[:, None], i3]

    out = np.concatenate([rgb, alpha[:, :, None]], 2).astype(np.uint8)
    out = out.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return out[:h, :w]


def vtf_to_image(path):
    """Decode the largest mip of frame 0 of a .vtf into an RGBA PIL image."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"VTF\0":
        raise ValueError("not a VTF file")
    minor = struct.unpack_from("<I", data, 8)[0]
    hsize = struct.unpack_from("<I", data, 12)[0]
    w, h, flags, frames, _first = struct.unpack_from("<HHIHH", data, 16)
    fmt = struct.unpack_from("<i", data, 52)[0]
    mips = max(1, data[56])
    lowfmt = struct.unpack_from("<i", data, 57)[0]
    lw, lh = data[61], data[62]
    depth = 1
    if minor >= 2:
        depth = max(1, struct.unpack_from("<H", data, 63)[0])
    faces = 6 if flags & 0x4000 else 1
    frames = max(1, frames)
    if minor >= 3:
        rc = struct.unpack_from("<I", data, 68)[0]
        off = None
        for k in range(rc):
            e = 80 + 8 * k
            if data[e:e + 3] == b"\x30\x00\x00":
                off = struct.unpack_from("<I", data, e + 4)[0]
        if off is None:
            raise ValueError("VTF has no image data resource")
    else:
        off = hsize
        if lowfmt != -1 and lw and lh:
            off += _img_size(lw, lh, lowfmt)
    for m in range(mips - 1, 0, -1):  # smaller mips are stored first
        off += _img_size(max(1, w >> m), max(1, h >> m), fmt) * frames * faces * depth
    raw = data[off:off + _img_size(w, h, fmt)]
    if len(raw) < _img_size(w, h, fmt):
        raise ValueError("VTF data is truncated")

    if fmt in (13, 14, 15, 20):
        arr = _decode_dxt(raw, w, h, fmt)
    else:
        bpp = _BPP.get(fmt)
        if bpp is None or fmt in (7, 22, 23, 24, 25, 26, 9, 10):
            raise ValueError("unsupported VTF format %s" % _FMT_NAMES.get(fmt, fmt))
        b = np.frombuffer(raw, dtype=np.uint8, count=w * h * bpp)
        full = np.full((h, w, 4), 255, np.uint8)
        if fmt == 0:
            full = b.reshape(h, w, 4)
        elif fmt == 1:
            full = b.reshape(h, w, 4)[..., ::-1]
        elif fmt == 2:
            full[..., :3] = b.reshape(h, w, 3)
        elif fmt == 3:
            full[..., :3] = b.reshape(h, w, 3)[..., ::-1]
        elif fmt == 5:
            full[..., :3] = b.reshape(h, w, 1)
        elif fmt == 6:
            t = b.reshape(h, w, 2)
            full[..., :3] = t[..., :1]
            full[..., 3] = t[..., 1]
        elif fmt == 8:
            full[..., :3] = 255
            full[..., 3] = b.reshape(h, w)
        elif fmt == 11:
            full = b.reshape(h, w, 4)[..., [1, 2, 3, 0]]
        elif fmt == 12:
            full = b.reshape(h, w, 4)[..., [2, 1, 0, 3]]
        elif fmt == 16:
            full[..., :3] = b.reshape(h, w, 4)[..., [2, 1, 0]]
        else:  # 16-bit packed formats
            v = np.frombuffer(raw, dtype="<u2", count=w * h).reshape(h, w).astype(np.int32)
            if fmt == 4:
                r, g, bl = (v >> 11) & 31, (v >> 5) & 63, v & 31
                full[..., 0], full[..., 1], full[..., 2] = r * 255 // 31, g * 255 // 63, bl * 255 // 31
            elif fmt == 17:
                bl, g, r = (v >> 11) & 31, (v >> 5) & 63, v & 31
                full[..., 0], full[..., 1], full[..., 2] = r * 255 // 31, g * 255 // 63, bl * 255 // 31
            elif fmt in (18, 21):
                bl, g, r = v & 31, (v >> 5) & 31, (v >> 10) & 31
                full[..., 0], full[..., 1], full[..., 2] = r * 255 // 31, g * 255 // 31, bl * 255 // 31
                if fmt == 21:
                    full[..., 3] = (v >> 15) * 255
            elif fmt == 19:
                bl, g, r, al = v & 15, (v >> 4) & 15, (v >> 8) & 15, (v >> 12) & 15
                full[..., 0], full[..., 1], full[..., 2], full[..., 3] = r * 17, g * 17, bl * 17, al * 17
        arr = np.ascontiguousarray(full).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def load_image(path):
    if path.lower().endswith(".vtf"):
        return vtf_to_image(path)
    return Image.open(path).convert("RGBA")


# --------------------------------------------------------------------------
#  SMD parsing + merging
# --------------------------------------------------------------------------
_NODE_RE = re.compile(r'^(-?\d+)\s+"([^"]*)"\s+(-?\d+)')


def _parse_vertex(s):
    p = s.split()
    parent = int(p[0])
    pos = (float(p[1]), float(p[2]), float(p[3]))
    nrm = (float(p[4]), float(p[5]), float(p[6]))
    uv = (float(p[7]), float(p[8]))
    links = []
    if len(p) > 9:
        for k in range(int(p[9])):
            links.append((int(p[10 + 2 * k]), float(p[11 + 2 * k])))
    if not links:
        links = [(parent, 1.0)]
    return pos, nrm, uv, links


def parse_smd(path):
    with open(path, "r", errors="replace") as f:
        lines = [l.strip() for l in f]
    nodes, skel, tris = [], {}, []
    section, first_time, cur_time = None, None, None
    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        i += 1
        if not l or l.startswith("//"):
            continue
        if section == "tris":
            if l == "end":
                section = None
                continue
            vs = []
            while len(vs) < 3 and i < n:
                vl = lines[i]
                i += 1
                if vl:
                    vs.append(vl)
            if len(vs) < 3:
                break
            tris.append((l, [_parse_vertex(x) for x in vs]))
            continue
        if l == "end":
            section = None
        elif l == "nodes":
            section = "nodes"
        elif l == "skeleton":
            section = "skel"
        elif l == "triangles":
            section = "tris"
        elif l == "vertexanimation":
            section = "skip"
        elif l.startswith("version"):
            pass
        elif section == "nodes":
            m = _NODE_RE.match(l)
            if m:
                nodes.append((int(m.group(1)), m.group(2), int(m.group(3))))
        elif section == "skel":
            p = l.split()
            if p[0] == "time":
                cur_time = int(p[1])
                if first_time is None:
                    first_time = cur_time
            elif cur_time == first_time and len(p) >= 7:
                skel[int(p[0])] = ([float(x) for x in p[1:4]], [float(x) for x in p[4:7]])
    return {"nodes": nodes, "skel": skel, "tris": tris}


class Mesh:
    pass


def merge_smds(parsed, scale=1.0, flip_v=False, flip_winding=True):
    bones, bidx = [], {}
    mats, midx = [], {}
    verts, vkey = [], {}
    tri_idx, tri_mat = [], []
    for d in parsed:
        id2name = {i: nm for i, nm, _ in d["nodes"]}
        remap = {}
        for i, nm, par in d["nodes"]:
            if nm not in bidx:
                pos, rot = d["skel"].get(i, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
                pn = id2name.get(par)
                bidx[nm] = len(bones)
                bones.append({"n": nm, "p": bidx.get(pn, -1) if par >= 0 else -1,
                              "pos": [c * scale for c in pos], "rot": list(rot)})
            remap[i] = bidx[nm]
        for mat, vs in d["tris"]:
            if mat not in midx:
                midx[mat] = len(mats)
                mats.append(mat)
            ids = []
            for pos, nrm, uv, links in vs:
                lk = {}
                for b, w in links:
                    rb = remap.get(b)
                    if rb is not None:
                        lk[rb] = lk.get(rb, 0.0) + w
                if not lk:
                    lk = {0: 1.0}
                top = sorted(lk.items(), key=lambda t: -t[1])[:4]
                tot = sum(w for _, w in top) or 1.0
                top = tuple((b, round(w / tot, 4)) for b, w in top)
                v = 1.0 - uv[1] if flip_v else uv[1]
                key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                       round(nrm[0], 4), round(nrm[1], 4), round(nrm[2], 4),
                       round(uv[0], 5), round(v, 5), top)
                vi = vkey.get(key)
                if vi is None:
                    vi = len(verts)
                    vkey[key] = vi
                    verts.append((pos, nrm, (uv[0], v), top))
                ids.append(vi)
            a, b, c = ids
            tri_idx.append((a, c, b) if flip_winding else (a, b, c))
            tri_mat.append(midx[mat])
    if not bones:
        raise ValueError("The SMD files contain no skeleton (nodes).")
    m = Mesh()
    nv = len(verts)
    m.bones, m.mats = bones, mats
    m.P = (np.array([v[0] for v in verts], np.float64).reshape(nv, 3) * scale)
    m.N = np.array([v[1] for v in verts], np.float64).reshape(nv, 3)
    m.UV = np.array([v[2] for v in verts], np.float64).reshape(nv, 2)
    m.SI = np.zeros((nv, 4), np.int32)
    m.SW = np.zeros((nv, 4), np.float64)
    for k, v in enumerate(verts):
        for j, (b, w) in enumerate(v[3]):
            m.SI[k, j], m.SW[k, j] = b, w
    m.tris = np.array(tri_idx, np.int64).reshape(-1, 3)
    m.tmat = np.array(tri_mat, np.int64)
    return m


# --------------------------------------------------------------------------
#  export (the exact layout the maze game's loader expects)
# --------------------------------------------------------------------------
def build_model_files(mesh, texfiles, mats_json):
    """returns (meta dict, gzip'd binary bytes)"""
    nv, nt = len(mesh.P), len(mesh.tris)
    if nv > 65535:
        raise ValueError("Too many vertices (%d) - the game uses 16-bit indices (max 65535)." % nv)
    if len(mesh.bones) > 255:
        raise ValueError("Too many bones (%d, max 255)." % len(mesh.bones))
    maxabs = max(float(np.abs(mesh.P).max()) if nv else 1.0, 1e-6)
    pscale = max(math.floor(min(1000.0, 32000.0 / maxabs) * 100) / 100.0, 0.01)
    pos16 = np.clip(np.round(mesh.P * pscale), -32767, 32767).astype("<i2")
    nrm8 = np.zeros((nv, 4), np.int8)
    nrm8[:, :3] = np.clip(np.round(mesh.N * 127), -127, 127).astype(np.int8)
    uv32 = mesh.UV.astype("<f4")
    si8 = mesh.SI.astype(np.uint8)
    w8 = np.round(mesh.SW * 255).astype(np.int32)
    for k in range(nv):  # make every vertex sum to exactly 255
        diff = 255 - int(w8[k].sum())
        if diff:
            w8[k, int(np.argmax(w8[k]))] += diff
    sw8 = np.clip(w8, 0, 255).astype(np.uint8)
    order = np.argsort(mesh.tmat, kind="stable")
    tris = mesh.tris[order]
    tm = mesh.tmat[order]
    idx16 = tris.reshape(-1).astype("<u2")
    groups = []
    if nt:
        start = 0
        for k in range(1, nt + 1):
            if k == nt or tm[k] != tm[start]:
                groups.append([start * 3, (k - start) * 3, int(tm[start])])
                start = k
    buf = bytearray()

    def put(arr, align=4):
        while len(buf) % align:
            buf.append(0)
        off = len(buf)
        buf.extend(arr.tobytes())
        return off

    meta = {"nv": nv, "ni": int(idx16.size), "pscale": pscale}
    meta["pos"] = put(pos16)
    meta["nrm"] = put(nrm8)
    meta["uv"] = put(uv32)
    meta["si"] = put(si8)
    meta["sw"] = put(sw8)
    meta["idx"] = put(idx16)
    meta["groups"] = groups
    meta["bones"] = [{"n": b["n"], "p": b["p"],
                      "pos": [round(c, 6) for c in b["pos"]],
                      "rot": [round(c, 8) for c in b["rot"]]} for b in mesh.bones]
    meta["texFiles"] = texfiles
    meta["mats"] = mats_json
    return meta, gzip.compress(bytes(buf), 9)


# --------------------------------------------------------------------------
#  preview renderer  (software, orthographic, painter's algorithm)
#  Uses the same coordinate change + lighting recipe as the game.
# --------------------------------------------------------------------------
def euler_zyx(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def bone_world_positions(bones):
    n = len(bones)
    Rw, Pw = [None] * n, [None] * n

    def get(i):
        if Rw[i] is not None:
            return
        b = bones[i]
        R, p = euler_zyx(*b["rot"]), np.array(b["pos"], float)
        if b["p"] >= 0:
            get(b["p"])
            Rw[i] = Rw[b["p"]] @ R
            Pw[i] = Pw[b["p"]] + Rw[b["p"]] @ p
        else:
            Rw[i], Pw[i] = R, p

    for i in range(n):
        get(i)
    return np.array(Pw)


def to_three(a):  # Source (x,y,z, Z-up)  ->  three.js (Y-up), same matrix as the game's MQ
    return np.stack([-a[:, 1], a[:, 2], -a[:, 0]], 1)


class Previewer:
    BG = (38, 40, 46)

    def __init__(self):
        self.mesh = None

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.T = to_three(mesh.P)
        self.TN = to_three(mesh.N)
        if len(self.T):
            self.center = (self.T.min(0) + self.T.max(0)) / 2
            self.radius = max(float(np.linalg.norm(self.T - self.center, axis=1).max()), 1e-6)
        else:
            self.center, self.radius = np.zeros(3), 1.0
        self.uvt = mesh.UV[mesh.tris].mean(1) if len(mesh.tris) else np.zeros((0, 2))
        self.BT = to_three(bone_world_positions(mesh.bones))
        self.mat_tex = [None] * len(mesh.mats)
        self.base = np.tile(np.array([255, 0, 255], np.float32), (len(mesh.tris), 1))

    def set_materials(self, info):
        """info[i] = ('tex', rgb ndarray) | ('color', (r,g,b)) | None"""
        m = self.mesh
        self.mat_tex = [None] * len(m.mats)
        base = np.tile(np.array([255, 0, 255], np.float32), (len(m.tris), 1))
        for mi, it in enumerate(info):
            if it is None:
                continue
            sel = np.nonzero(m.tmat == mi)[0]
            if not sel.size:
                continue
            if it[0] == "color":
                base[sel] = it[1]
            else:
                tx = it[1]
                self.mat_tex[mi] = tx
                th, tw = tx.shape[:2]
                u = np.mod(self.uvt[sel, 0], 1.0)
                v = np.mod(self.uvt[sel, 1], 1.0)
                xi = np.minimum((u * tw).astype(int), tw - 1)
                yi = np.minimum(((1 - v) * th).astype(int), th - 1)
                base[sel] = tx[yi, xi, :3]
        self.base = base

    def render(self, W, H, yaw, pitch, zoom, pan, wire=False, bones=False):
        img = Image.new("RGB", (W, H), self.BG)
        if self.mesh is None or not len(self.mesh.tris):
            return img
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        M = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        V = (self.T - self.center) @ M.T
        NV = self.TN @ M.T
        s = 0.46 * min(W, H) / self.radius * zoom
        SX = W / 2 + V[:, 0] * s + pan[0]
        SY = H / 2 - V[:, 1] * s + pan[1]
        tri = self.mesh.tris
        a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
        g = np.cross(b - a, c - a)
        front = g[:, 2] > 0  # CCW as seen by the camera = front face (three.js rule)
        fd = np.where(front, 1.0, -1.0)
        n = NV[tri].sum(1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        L = np.array([0.35, 0.55, 1.0])
        L /= np.linalg.norm(L)
        lam = np.clip((n @ L) * fd, 0, 1)
        shade = 0.19 + 0.30 + 0.70 * lam
        cols = np.clip(self.base * shade[:, None], 0, 255).astype(np.uint8)
        depth = (a[:, 2] + b[:, 2] + c[:, 2]) / 3
        order = np.argsort(depth, kind="stable")  # far (small z) first
        pts = np.stack([SX[tri], SY[tri]], 2).reshape(len(tri), 6)
        area = np.abs(g[:, 2]) * s * s / 2
        pts_o = pts[order].tolist()
        col_o = [tuple(x) for x in cols[order].tolist()]
        big_o = (area[order] > 500).tolist()
        mat_o = self.mesh.tmat[order].tolist()
        d = ImageDraw.Draw(img)
        for k in range(len(order)):
            p = pts_o[k]
            if wire:
                d.polygon(p, outline=(140, 190, 255))
                continue
            if big_o[k] and self.mat_tex[mat_o[k]] is not None:
                t = int(order[k])
                self._tex_tri(img, p, self.mat_tex[mat_o[k]], self.mesh.UV[tri[t]], float(shade[t]), W, H)
            else:
                d.polygon(p, fill=col_o[k], outline=col_o[k])
        if bones and len(self.BT):
            BV = (self.BT - self.center) @ M.T
            bx, by = W / 2 + BV[:, 0] * s + pan[0], H / 2 - BV[:, 1] * s + pan[1]
            for i, bn in enumerate(self.mesh.bones):
                if bn["p"] >= 0:
                    d.line([bx[bn["p"]], by[bn["p"]], bx[i], by[i]], fill=(255, 170, 60), width=2)
            for i in range(len(bx)):
                d.ellipse([bx[i] - 3, by[i] - 3, bx[i] + 3, by[i] + 3], fill=(255, 235, 120))
        return img

    @staticmethod
    def _tex_tri(img, p, tx, uv3, shade, W, H):
        xa, ya, xb, yb, xc, yc = p
        x0, x1 = max(int(math.floor(min(xa, xb, xc))), 0), min(int(math.ceil(max(xa, xb, xc))), W - 1)
        y0, y1 = max(int(math.floor(min(ya, yb, yc))), 0), min(int(math.ceil(max(ya, yb, yc))), H - 1)
        if x1 < x0 or y1 < y0:
            return
        den = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        if abs(den) < 1e-9:
            return
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        l0 = ((yb - yc) * (gx - xc) + (xc - xb) * (gy - yc)) / den
        l1 = ((yc - ya) * (gx - xc) + (xa - xc) * (gy - yc)) / den
        l2 = 1 - l0 - l1
        m = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
        if not m.any():
            return
        u = np.mod(l0 * uv3[0][0] + l1 * uv3[1][0] + l2 * uv3[2][0], 1.0)
        v = np.mod(l0 * uv3[0][1] + l1 * uv3[1][1] + l2 * uv3[2][1], 1.0)
        th, tw = tx.shape[:2]
        xi = np.minimum((u * tw).astype(int), tw - 1)
        yi = np.minimum(((1 - v) * th).astype(int), th - 1)
        col = np.clip(tx[yi, xi, :3].astype(np.float32) * shade, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(col, "RGB"), (x0, y0), Image.fromarray((m * 255).astype(np.uint8), "L"))


# --------------------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------------
CFG_PATH = os.path.join(os.path.expanduser("~"), ".playermodel_maker.json")


def load_cfg():
    try:
        with open(CFG_PATH, "r") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_cfg(cfg):
    try:
        with open(CFG_PATH, "w") as f:
            json.dump(cfg, f)
    except Exception:  # noqa: BLE001
        pass


def mat_stem(name):
    s = re.split(r"[\\/]", name.strip())[-1]
    return re.sub(r"\.(vtf|vmt|tga|png|jpg|jpeg|bmp)$", "", s, flags=re.I).lower()


def scan_textures(folder, max_depth=99):
    vtf, vmt, other = {}, {}, {}
    base_depth = folder.rstrip("\\/").count(os.sep)
    for dp, dn, fn in os.walk(folder):
        if dp.count(os.sep) - base_depth >= max_depth:
            dn[:] = []
        for f in fn:
            stem, ext = os.path.splitext(f)
            ext, k, full = ext.lower(), stem.lower(), os.path.join(dp, f)
            if ext == ".vtf":
                vtf.setdefault(k, full)
            elif ext == ".vmt":
                vmt.setdefault(k, full)
            elif ext in (".png", ".jpg", ".jpeg", ".tga", ".bmp"):
                other.setdefault(k, full)
    return vtf, vmt, other


def match_material(name, vtf, vmt, other):
    stem = mat_stem(name)
    if stem in vmt:
        try:
            with open(vmt[stem], errors="replace") as f:
                txt = f.read()
            m = re.search(r'"?\$basetexture"?\s+"?([^"\s]+)"?', txt, re.I)
            if m and mat_stem(m.group(1)) in vtf:
                return vtf[mat_stem(m.group(1))]
        except Exception:  # noqa: BLE001
            pass
    return vtf.get(stem) or other.get(stem)


def safe_name(s, fallback="x"):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._")
    return s or fallback


# --------------------------------------------------------------------------
#  GUI
# --------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Playermodel Maker")
        root.geometry("1260x800")
        root.minsize(1000, 640)
        self.cfg = load_cfg()
        self.smds = []
        self.matmap = {}
        self.mesh = None
        self.prev = Previewer()
        self.parse_cache, self.img_cache = {}, {}
        self.yaw, self.pitch, self.zoom, self.pan = math.pi, 0.12, 1.0, [0.0, 0.0]
        self.photo = None
        self._draw_job = None
        self._rebuild_job = None
        self._drag = None

        self.var_id = tk.StringVar()
        self.var_name = tk.StringVar()
        self.var_scale = tk.StringVar(value="1.0")
        self.var_flipw = tk.BooleanVar(value=True)
        self.var_flipv = tk.BooleanVar(value=False)
        self.var_alpha = tk.BooleanVar(value=True)
        self.var_maxtex = tk.StringVar(value="1024")
        self.var_out = tk.StringVar(value=self.cfg.get("out", ""))
        self.var_wire = tk.BooleanVar(value=False)
        self.var_bones = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Add your SMD files to begin.")
        self.build_ui()

    # ---- ui ---------------------------------------------------------------
    def build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("vista" if os.name == "nt" else "clam")
        except Exception:  # noqa: BLE001
            pass
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main, width=480)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        # 1 SMDs
        f1 = ttk.LabelFrame(left, text=" 1  SMD files  (merged into one model) ", padding=6)
        f1.pack(fill="x")
        self.lb = tk.Listbox(f1, height=5, selectmode="extended", activestyle="none", exportselection=False)
        self.lb.grid(row=0, column=0, sticky="nsew")
        bf = ttk.Frame(f1)
        bf.grid(row=0, column=1, padx=(6, 0), sticky="n")
        for t, c in (("Add...", self.add_smds), ("Remove", self.remove_smds),
                     ("Up", lambda: self.move_smd(-1)), ("Down", lambda: self.move_smd(1))):
            ttk.Button(bf, text=t, width=8, command=c).pack(pady=1)
        f1.columnconfigure(0, weight=1)

        # 2 materials
        f2 = ttk.LabelFrame(left, text=" 2  Materials  ->  textures ", padding=6)
        f2.pack(fill="both", expand=True, pady=(8, 0))
        self.tree = ttk.Treeview(f2, columns=("mat", "tex"), show="headings", height=9, selectmode="extended")
        self.tree.heading("mat", text="Material in SMD")
        self.tree.heading("tex", text="Texture (VTF / PNG) or colour")
        self.tree.column("mat", width=150)
        self.tree.column("tex", width=260)
        self.tree.tag_configure("missing", foreground="#c0392b")
        self.tree.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.tree.bind("<Double-1>", lambda e: self.set_tex())
        sb = ttk.Scrollbar(f2, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=4, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        ttk.Button(f2, text="Set VTF...", command=self.set_tex).grid(row=1, column=0, sticky="ew", pady=(5, 0))
        ttk.Button(f2, text="Set colour...", command=self.set_color).grid(row=1, column=1, sticky="ew", pady=(5, 0))
        ttk.Button(f2, text="Clear", command=self.clear_mat).grid(row=1, column=2, sticky="ew", pady=(5, 0))
        ttk.Button(f2, text="Auto-match folder...", command=self.auto_match_dialog).grid(row=1, column=3, sticky="ew", pady=(5, 0))
        for c in range(4):
            f2.columnconfigure(c, weight=1)
        f2.rowconfigure(0, weight=1)

        # 3 settings
        f3 = ttk.LabelFrame(left, text=" 3  Model settings ", padding=6)
        f3.pack(fill="x", pady=(8, 0))
        ttk.Label(f3, text="Model ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(f3, textvariable=self.var_id, width=18).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(f3, text="Display name").grid(row=0, column=2, sticky="w")
        ttk.Entry(f3, textvariable=self.var_name, width=18).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Label(f3, text="Scale").grid(row=1, column=0, sticky="w", pady=(4, 0))
        e = ttk.Entry(f3, textvariable=self.var_scale, width=8)
        e.grid(row=1, column=1, sticky="w", padx=4, pady=(4, 0))
        e.bind("<Return>", lambda ev: self.schedule_rebuild())
        e.bind("<FocusOut>", lambda ev: self.schedule_rebuild())
        ttk.Label(f3, text="Max texture size").grid(row=1, column=2, sticky="w", pady=(4, 0))
        ttk.Combobox(f3, textvariable=self.var_maxtex, width=8, state="readonly",
                     values=("256", "512", "1024", "2048", "original")).grid(row=1, column=3, sticky="w", padx=4, pady=(4, 0))
        ttk.Checkbutton(f3, text="Flip triangle winding (Source -> three.js)", variable=self.var_flipw,
                        command=self.schedule_rebuild).grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Checkbutton(f3, text="Flip V (texture upside-down)", variable=self.var_flipv,
                        command=self.schedule_rebuild).grid(row=3, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(f3, text="Drop texture alpha (VTF alpha is usually a specular mask)", variable=self.var_alpha
                        ).grid(row=4, column=0, columnspan=4, sticky="w")
        f3.columnconfigure(1, weight=1)
        f3.columnconfigure(3, weight=1)

        # 4 export
        f4 = ttk.LabelFrame(left, text=" 4  Export ", padding=6)
        f4.pack(fill="x", pady=(8, 0))
        ttk.Label(f4, text="Output folder (playermodels repo)").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(f4, textvariable=self.var_out).grid(row=1, column=0, sticky="ew")
        ttk.Button(f4, text="Browse...", command=self.browse_out).grid(row=1, column=1, padx=(4, 0))
        ttk.Button(f4, text="Export playermodel", command=self.export_model).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0), ipady=4)
        f4.columnconfigure(0, weight=1)

        # preview
        self.canvas = tk.Canvas(right, bg="#262830", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.request_draw())
        self.canvas.bind("<ButtonPress-1>", lambda e: self._start_drag(e, "rot"))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._start_drag(e, "pan"))
        self.canvas.bind("<ButtonPress-2>", lambda e: self._start_drag(e, "pan"))
        for ev in ("<B1-Motion>", "<B3-Motion>", "<B2-Motion>"):
            self.canvas.bind(ev, self._do_drag)
        self.canvas.bind("<MouseWheel>", lambda e: self._zoom(1.1 if e.delta > 0 else 1 / 1.1))
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1 / 1.1))
        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Button(bar, text="Reset view", command=self.reset_view).pack(side="left")
        for t, yw in (("Front", math.pi), ("Back", 0.0), ("Left", math.pi / 2), ("Right", -math.pi / 2)):
            ttk.Button(bar, text=t, width=6, command=lambda y=yw: self.set_yaw(y)).pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="Wireframe", variable=self.var_wire, command=self.request_draw).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(bar, text="Skeleton", variable=self.var_bones, command=self.request_draw).pack(side="left", padx=6)
        self.info = ttk.Label(bar, text="")
        self.info.pack(side="right")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w", padding=(6, 2)).pack(fill="x", side="bottom")

    # ---- smd list -----------------------------------------------------------
    def add_smds(self):
        ps = filedialog.askopenfilenames(title="Select SMD files", initialdir=self.cfg.get("smd_dir") or None,
                                         filetypes=[("Studiomdl Data", "*.smd"), ("All files", "*.*")])
        if not ps:
            return
        self.cfg["smd_dir"] = os.path.dirname(ps[0])
        save_cfg(self.cfg)
        for p in ps:
            if p not in self.smds:
                self.smds.append(p)
        if not self.var_id.get().strip():
            stem = safe_name(os.path.splitext(os.path.basename(ps[0]))[0].lower())
            self.var_id.set(stem)
            self.var_name.set(stem.replace("_", " ").title())
        self.refresh_listbox()
        self.rebuild()

    def remove_smds(self):
        for i in sorted(self.lb.curselection(), reverse=True):
            del self.smds[i]
        self.refresh_listbox()
        self.rebuild()

    def move_smd(self, d):
        sel = self.lb.curselection()
        if len(sel) != 1:
            return
        i, j = sel[0], sel[0] + d
        if 0 <= j < len(self.smds):
            self.smds[i], self.smds[j] = self.smds[j], self.smds[i]
            self.refresh_listbox()
            self.lb.selection_set(j)
            self.rebuild()

    def refresh_listbox(self):
        self.lb.delete(0, "end")
        for p in self.smds:
            self.lb.insert("end", os.path.basename(p))

    # ---- build / materials -------------------------------------------------
    def schedule_rebuild(self):
        if self._rebuild_job:
            self.root.after_cancel(self._rebuild_job)
        self._rebuild_job = self.root.after(250, self.rebuild)

    def get_parsed(self, p):
        mt = os.path.getmtime(p)
        c = self.parse_cache.get(p)
        if c and c[0] == mt:
            return c[1]
        d = parse_smd(p)
        self.parse_cache[p] = (mt, d)
        return d

    def scale_value(self):
        try:
            v = float(self.var_scale.get().replace(",", "."))
            return v if v > 0 else 1.0
        except ValueError:
            return 1.0

    def rebuild(self):
        self._rebuild_job = None
        if not self.smds:
            self.mesh = None
            self.prev.mesh = None
            self.refresh_tree()
            self.request_draw()
            self.info.config(text="")
            return
        try:
            parsed = [self.get_parsed(p) for p in self.smds]
            self.mesh = merge_smds(parsed, self.scale_value(), self.var_flipv.get(), self.var_flipw.get())
        except Exception as ex:  # noqa: BLE001
            self.status.set("SMD error: %s" % ex)
            messagebox.showerror("Playermodel Maker", "Could not read the SMD files:\n\n%s" % ex)
            return
        # auto-guess textures for new materials from folders next to the SMDs
        new = [m for m in self.mesh.mats if m not in self.matmap]
        if new:
            vtf, vmt, other = {}, {}, {}
            for d in sorted({os.path.dirname(p) for p in self.smds}):
                a, b, c = scan_textures(d, max_depth=3)
                for tgt, src in ((vtf, a), (vmt, b), (other, c)):
                    for k, v in src.items():
                        tgt.setdefault(k, v)
            for m in new:
                hit = match_material(m, vtf, vmt, other)
                if hit:
                    self.matmap[m] = {"type": "tex", "path": hit}
        self.prev.set_mesh(self.mesh)
        self.refresh_tree()
        self.update_preview_mats()
        m = self.mesh
        self.info.config(text="%d verts   %d tris   %d bones   %d materials" % (len(m.P), len(m.tris), len(m.bones), len(m.mats)))
        self.status.set("Merged %d SMD file(s)." % len(self.smds))

    def refresh_tree(self):
        sel = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        if not self.mesh:
            return
        for name in self.mesh.mats:
            mm = self.matmap.get(name)
            if mm and mm["type"] == "tex":
                txt, tag = os.path.basename(mm["path"]), ()
            elif mm and mm["type"] == "color":
                txt, tag = "colour " + mm["c"], ()
            else:
                txt, tag = "- not set -", ("missing",)
            self.tree.insert("", "end", iid=name, values=(name, txt), tags=tag)
        for s in sel:
            if self.tree.exists(s):
                self.tree.selection_add(s)

    def sel_mats(self):
        s = list(self.tree.selection())
        return s if s else list(self.tree.get_children()[:0])

    def set_tex(self):
        names = self.sel_mats()
        if not names:
            messagebox.showinfo("Playermodel Maker", "Select one or more materials in the table first.")
            return
        p = filedialog.askopenfilename(title="Texture for: " + ", ".join(names)[:80],
                                       initialdir=self.cfg.get("tex_dir") or None,
                                       filetypes=[("Textures", "*.vtf *.png *.jpg *.jpeg *.tga *.bmp"), ("All files", "*.*")])
        if not p:
            return
        self.cfg["tex_dir"] = os.path.dirname(p)
        save_cfg(self.cfg)
        for n in names:
            self.matmap[n] = {"type": "tex", "path": p}
        self.refresh_tree()
        self.update_preview_mats()

    def set_color(self):
        names = self.sel_mats()
        if not names:
            messagebox.showinfo("Playermodel Maker", "Select one or more materials in the table first.")
            return
        c = colorchooser.askcolor(title="Flat colour")
        if not c or not c[1]:
            return
        for n in names:
            self.matmap[n] = {"type": "color", "c": c[1]}
        self.refresh_tree()
        self.update_preview_mats()

    def clear_mat(self):
        for n in self.sel_mats():
            self.matmap.pop(n, None)
        self.refresh_tree()
        self.update_preview_mats()

    def auto_match_dialog(self):
        if not self.mesh:
            messagebox.showinfo("Playermodel Maker", "Add SMD files first.")
            return
        d = filedialog.askdirectory(title="Folder containing your .vtf / .vmt files (searched recursively)",
                                    initialdir=self.cfg.get("tex_dir") or None)
        if not d:
            return
        self.cfg["tex_dir"] = d
        save_cfg(self.cfg)
        self.status.set("Scanning %s ..." % d)
        self.root.update_idletasks()
        vtf, vmt, other = scan_textures(d)
        names = self.sel_mats() or list(self.mesh.mats)
        hit = 0
        for n in names:
            p = match_material(n, vtf, vmt, other)
            if p:
                self.matmap[n] = {"type": "tex", "path": p}
                hit += 1
        self.refresh_tree()
        self.update_preview_mats()
        self.status.set("Auto-match: %d of %d materials matched (%d VTFs found)." % (hit, len(names), len(vtf)))

    # ---- textures / preview -------------------------------------------------
    def get_image(self, path):
        mt = os.path.getmtime(path)
        c = self.img_cache.get(path)
        if c and c[0] == mt:
            return c[1], c[2]
        img = load_image(path)
        rgb = np.asarray(img.convert("RGB"))
        self.img_cache[path] = (mt, img, rgb)
        return img, rgb

    def update_preview_mats(self):
        if not self.mesh:
            return
        info, errs = [], []
        for name in self.mesh.mats:
            mm = self.matmap.get(name)
            if mm and mm["type"] == "tex":
                try:
                    info.append(("tex", self.get_image(mm["path"])[1]))
                except Exception as ex:  # noqa: BLE001
                    errs.append("%s: %s" % (os.path.basename(mm["path"]), ex))
                    info.append(None)
            elif mm and mm["type"] == "color":
                h = mm["c"].lstrip("#")
                info.append(("color", tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))))
            else:
                info.append(None)
        self.prev.set_materials(info)
        if errs:
            self.status.set("Texture problem - " + "; ".join(errs))
        self.request_draw()

    def request_draw(self):
        if self._draw_job is None:
            self._draw_job = self.root.after(12, self.draw)

    def draw(self):
        self._draw_job = None
        W, H = self.canvas.winfo_width(), self.canvas.winfo_height()
        if W < 20 or H < 20:
            return
        img = self.prev.render(W, H, self.yaw, self.pitch, self.zoom, self.pan, self.var_wire.get(), self.var_bones.get())
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        if not self.mesh:
            self.canvas.create_text(W / 2, H / 2, fill="#8a8f9c", font=("Segoe UI", 13),
                                    text="Preview\n\nAdd SMD files on the left")

    def _start_drag(self, e, mode):
        self._drag = (mode, e.x, e.y)

    def _do_drag(self, e):
        if not self._drag:
            return
        mode, x, y = self._drag
        dx, dy = e.x - x, e.y - y
        if mode == "rot":
            self.yaw += dx * 0.01
            self.pitch = max(-1.45, min(1.45, self.pitch + dy * 0.01))
        else:
            self.pan[0] += dx
            self.pan[1] += dy
        self._drag = (mode, e.x, e.y)
        self.request_draw()

    def _zoom(self, f):
        self.zoom = max(0.2, min(12.0, self.zoom * f))
        self.request_draw()

    def set_yaw(self, y):
        self.yaw, self.pitch = y, 0.0
        self.request_draw()

    def reset_view(self):
        self.yaw, self.pitch, self.zoom, self.pan = math.pi, 0.12, 1.0, [0.0, 0.0]
        self.request_draw()

    # ---- export ----------------------------------------------------------
    def browse_out(self):
        d = filedialog.askdirectory(title="Output folder (your local playermodels repo)", initialdir=self.var_out.get() or None)
        if d:
            self.var_out.set(d)
            self.cfg["out"] = d
            save_cfg(self.cfg)

    def export_model(self):
        try:
            self._export()
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Playermodel Maker", "Export failed:\n\n%s" % ex)

    def _export(self):
        if not self.mesh or not len(self.mesh.tris):
            messagebox.showwarning("Playermodel Maker", "Nothing to export - add SMD files that contain triangles.")
            return
        mid = self.var_id.get().strip()
        if not re.fullmatch(r"[\w-]+", mid):
            messagebox.showwarning("Playermodel Maker", "Model ID may only contain letters, digits, _ and -.")
            return
        if mid == "mike":
            messagebox.showwarning("Playermodel Maker", "'mike' is the built-in Combine Soldier - pick another ID.")
            return
        out = self.var_out.get().strip()
        if not out:
            self.browse_out()
            out = self.var_out.get().strip()
            if not out:
                return
        folder = os.path.join(out, mid)
        if os.path.exists(os.path.join(folder, "meta.json")):
            if not messagebox.askyesno("Playermodel Maker", "'%s' already exists in that folder. Overwrite it?" % mid):
                return
        os.makedirs(folder, exist_ok=True)
        self.cfg["out"] = out
        save_cfg(self.cfg)

        maxtex = self.var_maxtex.get()
        limit = None if maxtex == "original" else int(maxtex)
        used = set(self.mesh.tmat.tolist())
        texfiles, texkey, mats_json, names_used, missing = [], {}, [], set(), []
        for mi, name in enumerate(self.mesh.mats):
            mm = self.matmap.get(name)
            if mi not in used:
                mats_json.append({"k": "color", "c": "#808080"})
            elif mm and mm["type"] == "tex":
                p = mm["path"]
                if p not in texkey:
                    img = self.get_image(p)[0]
                    img = img.convert("RGB") if self.var_alpha.get() else img
                    if limit and max(img.size) > limit:
                        f = limit / max(img.size)
                        img = img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.LANCZOS)
                    base = safe_name(os.path.splitext(os.path.basename(p))[0], "tex")
                    fn, k = base + ".png", 2
                    while fn in names_used:
                        fn, k = "%s_%d.png" % (base, k), k + 1
                    names_used.add(fn)
                    img.save(os.path.join(folder, fn), optimize=True)
                    texkey[p] = len(texfiles)
                    texfiles.append(fn)
                mats_json.append({"k": "tex", "t": texkey[p]})
            elif mm and mm["type"] == "color":
                mats_json.append({"k": "color", "c": mm["c"]})
            else:
                mats_json.append({"k": "color", "c": "#ff00ff"})
                missing.append(name)

        meta, binary = build_model_files(self.mesh, texfiles, mats_json)
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, separators=(",", ":"))
        with open(os.path.join(folder, "model.bin.gz"), "wb") as f:
            f.write(binary)

        mj = os.path.join(out, "models.json")
        data = {"models": []}
        if os.path.exists(mj):
            try:
                with open(mj) as f:
                    data = json.load(f)
                if not isinstance(data.get("models"), list):
                    data["models"] = []
            except Exception:  # noqa: BLE001
                data = {"models": []}
        entry = {"id": mid, "name": self.var_name.get().strip() or mid,
                 "src": ", ".join(os.path.basename(p) for p in self.smds)}
        data["models"] = [e for e in data["models"] if not (isinstance(e, dict) and e.get("id") == mid)] + [entry]
        with open(mj, "w") as f:
            json.dump(data, f, indent=1)

        msg = ("Exported '%s'\n\n%s\n  meta.json, model.bin.gz, %d texture(s)\n%s (updated)\n\n"
               "%d verts, %d tris, %d bones.\n\nUpload the '%s' folder and models.json to your playermodels repo."
               % (mid, folder, len(texfiles), mj, len(self.mesh.P), len(self.mesh.tris), len(self.mesh.bones), mid))
        if missing:
            msg += "\n\nNo texture set for: " + ", ".join(missing) + "  (exported as magenta)"
        self.status.set("Exported %s -> %s" % (mid, folder))
        messagebox.showinfo("Playermodel Maker", msg)


def main():
    try:
        ctypes_ok = os.name == "nt"
        if ctypes_ok:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        pass
    root = tk.Tk()

    def on_err(exc, val, tb):
        messagebox.showerror("Playermodel Maker", "".join(traceback.format_exception(exc, val, tb))[-1800:])

    root.report_callback_exception = on_err
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
