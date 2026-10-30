"""
LOTUS, Phase 4: the camera student.

A frozen vision foundation model encodes three front cameras; a small head maps the image features,
the car's speed and its navigation route to the driving command (target speed, sideways position).
Auxiliary heads predict the signal colour and a bird's-eye occupancy grid, which the teacher knows
exactly (privileged), to make the image features learn what matters.

  python carla/student.py encode --data runs/phase4/data --encoder dinov3
  python carla/student.py train  --data runs/phase4/data --encoder dinov3 --out runs/phase4/student.pt

Encoders: dinov3 (facebook/dinov3-vits16-pretrain-lvd1689m, gated on Hugging Face: accept the licence and
`huggingface-cli login`), vjepa2 (facebook/vjepa2-vitl-fpc64-256, video model, uses the last 4 frames),
dinov2 (torch.hub, open), tiny (random small CNN, for smoke tests only).
"""
import argparse, glob, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAMS = ["left", "front", "right"]
BEV = 24          # occupancy grid side (cells of 1 m, 24 m ahead x 24 m wide)
NAV = 1 + 20 + 1  # speed, 10 route waypoints (x, y), distance to goal


# ---------------------------------------------------------------- encoders (frozen)
class Encoder(nn.Module):
    def __init__(self, name, device):
        super().__init__()
        self.name, self.device = name, device
        if name == "dinov3":
            from transformers import AutoModel
            self.m = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m"); self.dim = 2 * self.m.config.hidden_size
        elif name == "vjepa2":
            from transformers import AutoModel
            self.m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256"); self.dim = self.m.config.hidden_size
        elif name == "dinov2":
            self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"); self.dim = 2 * 384
        elif name == "tiny":
            torch.manual_seed(0)
            self.m = nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten())
            self.dim = 64 * 16
        else: raise ValueError(name)
        self.m.eval().to(device)
        for p in self.m.parameters(): p.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1); self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, imgs):
        """imgs: uint8 tensor (B, H, W, 3) for image models, or (B, T, H, W, 3) for vjepa2 -> (B, dim)"""
        if self.name == "vjepa2":
            x = imgs.to(self.device).float().div(255).permute(0, 1, 4, 2, 3)            # B T C H W
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), size=(256, 256), mode="bilinear", align_corners=False)
            x = ((x - self.mean) / self.std).view(B, T, 3, 256, 256)
            return self.m(pixel_values_videos=x).last_hidden_state.mean(1)
        x = imgs.to(self.device).float().div(255).permute(0, 3, 1, 2)
        size = 224 if self.name != "dinov3" else 224
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        if self.name == "dinov3":
            h = self.m(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        if self.name == "dinov2":
            h = self.m.forward_features(x)
            return torch.cat([h["x_norm_clstoken"], h["x_norm_patchtokens"].mean(1)], 1)
        return self.m(x)


# ---------------------------------------------------------------- the trainable head
class Head(nn.Module):
    def __init__(self, feat_dim, hidden=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU())              # shared per camera
        inp = 256 * len(CAMS) * 2 + NAV + 2                                            # this and the previous frame, nav, previous command
        self.trunk = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU())
        self.act = nn.Linear(hidden, 2); self.light = nn.Linear(hidden, 4); self.bev = nn.Linear(hidden, BEV * BEV)

    def forward(self, f, f_prev, nav, prev_act):
        B = f.shape[0]
        z = torch.cat([self.proj(f).flatten(1), self.proj(f_prev).flatten(1), nav, prev_act], 1)
        h = self.trunk(z)
        return torch.tanh(self.act(h)), self.light(h), self.bev(h).view(B, BEV, BEV)


def nav_vec(nav):
    return np.array([nav["speed"] / 10] + list(nav["waypoints"]) + [nav["toGoal"]], np.float32)


# ---------------------------------------------------------------- dataset on disk (written by bridge.py)
def episodes(data):
    return sorted(glob.glob(os.path.join(data, "ep_*")))


def load_episode(ep):
    meta = np.load(os.path.join(ep, "meta.npz"))
    return {k: meta[k] for k in meta.files}


def read_frame(ep, i):
    from PIL import Image
    return np.stack([np.asarray(Image.open(os.path.join(ep, f"{i:05d}_{c}.jpg")).convert("RGB")) for c in CAMS])


def encode(a):
    dev = torch.device(a.device); enc = Encoder(a.encoder, dev)
    for ep in episodes(a.data):
        out = os.path.join(ep, f"feat_{a.encoder}.npy")
        if os.path.exists(out): continue
        n = len(load_episode(ep)["label"]); feats = np.zeros((n, len(CAMS), enc.dim), np.float16)
        frames = [read_frame(ep, i) for i in range(n)]
        for i in range(n):
            if a.encoder == "vjepa2":   # clip of the last 4 frames per camera
                idx = [max(0, i - k) for k in (3, 2, 1, 0)]
                clips = torch.from_numpy(np.stack([np.stack([frames[j][c] for j in idx]) for c in range(len(CAMS))]))
                feats[i] = enc(clips).cpu().numpy()
            else: feats[i] = enc(torch.from_numpy(frames[i])).cpu().numpy()
        np.save(out, feats)
        print(json.dumps({"encoded": os.path.basename(ep), "frames": n}), flush=True)


def build_arrays(data, encoder):
    F_, P_, N_, A_, Y_, L_, B_ = [], [], [], [], [], [], []
    for ep in episodes(data):
        fp = os.path.join(ep, f"feat_{encoder}.npy")
        if not os.path.exists(fp): continue
        f = np.load(fp).astype(np.float32); m = load_episode(ep)
        prev = np.concatenate([f[:1], f[:-1]])
        pa = np.concatenate([np.zeros((1, 2), np.float32), m["executed"][:-1]])
        F_.append(f); P_.append(prev); N_.append(m["nav"]); A_.append(pa); Y_.append(m["label"]); L_.append(m["light"]); B_.append(m["bev"])
    cat = lambda xs: np.concatenate(xs)
    return cat(F_), cat(P_), cat(N_), cat(A_), cat(Y_), cat(L_), cat(B_)


def train(a):
    dev = torch.device(a.device)
    Fe, Pe, Na, Pa, Y, L, Bv = build_arrays(a.data, a.encoder)
    n = len(Y); print(json.dumps({"samples": n}), flush=True)
    T = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=dev)
    Fe, Pe, Na, Pa, Y, L, Bv = T(Fe), T(Pe), T(Na), T(Pa), T(Y), T(L, torch.long), T(Bv.astype(np.float32) / 255)
    head = Head(Fe.shape[-1]).to(dev)
    if a.init and os.path.exists(a.init): head.load_state_dict(torch.load(a.init, map_location=dev)["head"])
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    perm = torch.randperm(n, device=dev); nv = max(64, n // 10); va, tr = perm[:nv], perm[nv:]
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        head.train(); idx = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(idx), 256):
            j = idx[i:i + 256]
            # light augmentation: feature noise and occasional dropped camera
            fe = Fe[j] + 0.02 * torch.randn_like(Fe[j]); fe = fe * (torch.rand(len(j), len(CAMS), 1, device=dev) > 0.05)
            act, lg, bev = head(fe, Pe[j], Na[j], Pa[j])
            loss = ((act - Y[j]) ** 2 * torch.tensor([2.0, 1.0], device=dev)).mean() + a.aux * (F.cross_entropy(lg, L[j]) + F.binary_cross_entropy_with_logits(bev, Bv[j]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        sched.step(); head.eval()
        with torch.no_grad():
            act, lg, _ = head(Fe[va], Pe[va], Na[va], Pa[va])
            vl = float(((act - Y[va]) ** 2).mean()); acc = float((lg.argmax(1) == L[va]).float().mean())
        if vl < best: best, best_state = vl, {k: v.clone() for k, v in head.state_dict().items()}
        if ep % max(1, a.epochs // 8) == 0 or ep == a.epochs - 1: print(json.dumps({"epoch": ep, "val_mse": round(vl, 4), "light_acc": round(acc, 3)}), flush=True)
    torch.save({"head": best_state, "encoder": a.encoder, "feat_dim": int(Fe.shape[-1])}, a.out)
    print(json.dumps({"saved": a.out, "best_val_mse": round(best, 4)}))


class StudentDriver:
    """used by bridge.py to drive from camera images"""
    def __init__(self, ckpt, device="cuda"):
        ck = torch.load(ckpt, map_location=device); self.dev = torch.device(device)
        self.enc = Encoder(ck["encoder"], self.dev); self.head = Head(ck["feat_dim"]).to(self.dev); self.head.load_state_dict(ck["head"]); self.head.eval()
        self.prev = None; self.prev_act = np.zeros(2, np.float32); self.hist = []

    def reset(self): self.prev = None; self.prev_act[:] = 0; self.hist = []

    @torch.no_grad()
    def act(self, imgs, nav):
        if self.enc.name == "vjepa2":
            self.hist = (self.hist + [imgs])[-4:]; h = [self.hist[0]] * (4 - len(self.hist)) + self.hist
            f = self.enc(torch.from_numpy(np.stack([np.stack([x[c] for x in h]) for c in range(len(CAMS))]))).unsqueeze(0)
        else: f = self.enc(torch.from_numpy(imgs)).unsqueeze(0)
        p = self.prev if self.prev is not None else f
        a, _, _ = self.head(f.float(), p.float(), torch.tensor(nav_vec(nav)[None], device=self.dev), torch.tensor(self.prev_act[None], device=self.dev))
        self.prev = f; self.prev_act = a[0].cpu().numpy()
        return self.prev_act.tolist()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "train"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True); s.add_argument("--encoder", default="dinov3")
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if name == "train":
            s.add_argument("--out", required=True); s.add_argument("--init", default=None); s.add_argument("--epochs", type=int, default=30)
            s.add_argument("--lr", type=float, default=3e-4); s.add_argument("--aux", type=float, default=0.2)
    a = p.parse_args()
    encode(a) if a.cmd == "encode" else train(a)
