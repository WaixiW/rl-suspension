# LSTM State Stabilization Plan for Streaming Speech Separation

Goal: eliminate long-stream performance degradation of the LSTM separation backbone
without destroying the speaker-to-channel binding encoded in the recurrent state.

Two approaches, executed in order:

- **Approach 1 (training-free):** clip / decay / re-standardize the state at inference.
- **Approach 2 (light trainable):** a small state-refresh network `g(s_drift) -> s_fresh`.

Both depend on **Phase 0 (instrumentation)**, so do that first regardless.

Assumed streaming interface (adapt names to the real model):

```python
states = model.init_states()                      # list over layers of (h, c)
for frame in stream:                              # 10 ms hop
    feats = model.encoder(frame)
    out, states = model.lstm_step(feats, states)
    y = model.decoder(out)                        # [2, hop] separated channels
```

---

## Phase 0 — State trajectory tracking (diagnosis)

### 0.1 Tracker

Record per-layer state statistics every ~1 s (100 frames at 10 ms hop).
Keep full per-dimension dumps at a coarser interval to limit memory.

```python
class StateTracker:
    def __init__(self, log_every=100, dump_every=1000):
        self.log_every, self.dump_every = log_every, dump_every
        self.records = []
        self.t = 0

    def update(self, states):
        if self.t % self.log_every == 0:
            rec = {"t": self.t}
            for l, (h, c) in enumerate(states):
                rec[f"h{l}_norm"]   = float(h.norm())
                rec[f"c{l}_norm"]   = float(c.norm())
                rec[f"c{l}_maxabs"] = float(c.abs().max())
                if self.t % self.dump_every == 0:
                    rec[f"c{l}_vals"] = c.detach().squeeze().cpu().numpy().copy()
            self.records.append(rec)
        self.t += 1

    def save(self, path):
        np.savez(path, records=self.records)
```

Insert `tracker.update(states)` right after `lstm_step` in the streaming loop.

### 0.2 Reference statistics (healthy distribution)

Run the model over M (>= 200) training-length utterances and collect states from
the *second half* of each utterance (steady state, past the cold-start transient):

```python
all_c[l] = concat of c-states, shape [N_samples, hidden]

ref = {
    "mu":     all_c[l].mean(0),          # per-dim mean
    "sigma":  all_c[l].std(0),           # per-dim std
    "c_p999": percentile(|all_c[l]|, 99.9, axis=0),   # per-dim clip bound
    "norm_p99": percentile(norm(all_c[l]), 99),       # per-layer norm bound
}
save ref per layer -> reference_stats.npz
```

### 0.3 Long-run drift analysis

Run on >= 30 min of continuous mixture audio (concatenate utterances so you also
have references for quality measurement), with the tracker on.

Analysis to produce:

1. **Norm-vs-time plot** per layer (`c_norm`, `h_norm`). Compare against `norm_p99`.
2. **Runaway-dimension detection:** for each cell dim, fit a linear slope of
   `|c_dim|` over time; flag dims with slope significantly > 0.

```python
slopes[l] = linfit(t_minutes, abs(c_vals[l]))        # per dim
runaway_dims[l] = where(slopes[l] > slope_thresh)    # e.g. > 0.05/min
```

3. **Quality-vs-time:** SI-SDR computed in 1-min buckets over the long stream.
   Correlate the knee of the SI-SDR curve with the state drift curves.

### 0.4 Decision gate

- Few dims growing ~linearly (saturated forget gates / integrators)
  -> Approach 1 will very likely suffice; prioritize per-dim decay + clipping.
- Whole distribution shifting (mean/std drift, norms roughly stable)
  -> Approach 1 re-standardization first; expect to need Approach 2.

---

## Phase 0b — Evaluating state degradation (three levels + calibration)

Degradation can be measured at three levels. State-level metrics are cheap but
only indicate "state left the training distribution", not "output got worse"
(false positives: large-but-harmless drift, e.g. into tanh saturation; false
negatives: subtle distributional shift with stable norms). End-to-end SI-SDR is
ground truth but needs references. The divergence probe sits in between:
reference-free and functional. **Calibrate levels 1–2 against level 3 once,
offline; then use levels 1–2 as the cheap monitors.**

### Level 1 — Direct state metrics (per frame, ~free)

```python
class StateHealthMonitor:
    def __init__(self, ref):                 # ref from Phase 0.2
        self.mu, self.sigma = ref.mu, ref.sigma
        self.norm_p99 = ref.norm_p99

    def __call__(self, states, gates=None):
        m = {}
        for l, (h, c) in enumerate(states):
            m[f"c{l}_norm_ratio"] = float(c.norm()) / self.norm_p99[l]
            z = (c - self.mu[l]) / (self.sigma[l] + 1e-6)
            m[f"c{l}_n_outlier"]  = int((z.abs() > 4).sum())   # dims with |z|>4
            m[f"c{l}_mahal"]      = float((z ** 2).mean().sqrt())  # diag Mahalanobis
            if gates is not None:              # needs custom cell / forward hook
                m[f"f{l}_saturation"] = float((gates[l].f > 0.99).float().mean())
        return m
```

Note: forget-gate saturation requires access to gate activations (custom LSTM
cell or a forward hook); skip it if the model uses a fused cuDNN LSTM.

### Level 2 — Fresh-restart divergence probe (reference-free, periodic)

Compare the long-running instance's output to a fresh zero-state burn-in over
the same recent input. Isolates the state as the cause by construction; works
on real unlabeled audio in deployment.

```python
PROBE_EVERY = 60 * 100                  # every 60 s (frames @ 10 ms)
W = 4                                   # probe window, seconds

ring_in  = RingBuffer(seconds=W)        # recent input frames
ring_out = RingBuffer(seconds=W)        # recent live outputs [2, T]

for t, frame in enumerate(stream):
    ring_in.push(frame)
    out, states = model.lstm_step(feats(frame), states)
    ring_out.push(decode(out))

    if t % PROBE_EVERY == 0 and t > 0:
        y_fresh, _ = model.run_chunk(ring_in.contents(), model.init_states())
        y_live     = ring_out.contents()
        # min over permutations handles the fresh run's arbitrary channel order;
        # score only the tail (skip the fresh run's cold-start transient)
        tail = slice(int(0.5 * W * fs), None)
        d_id = dist(y_fresh[0][tail], y_live[0][tail]) + dist(y_fresh[1][tail], y_live[1][tail])
        d_sw = dist(y_fresh[0][tail], y_live[1][tail]) + dist(y_fresh[1][tail], y_live[0][tail])
        divergence = min(d_id, d_sw)    # dist = e.g. -SI-SDR or log-spectral L2
        log(t, divergence)
```

Small, flat `divergence` over stream time => drifted state is functionally
harmless regardless of norms. Growing `divergence` => genuine degradation.
Cost: one extra forward pass over W seconds per probe interval.

### Level 3 — End-to-end SI-SDR vs stream time (ground truth, offline)

Requires long synthesized mixtures with references (Phase 0.3 material).

```python
BUCKET = 60 * fs                        # 1-minute buckets
y = streaming_inference(mix)            # full long stream, fix under test ON/OFF
for b in range(n_buckets):
    seg = slice(b * BUCKET, (b + 1) * BUCKET)
    sisdr[b] = mean over utterances of
               max over permutations of si_sdr(y[:, seg], ref[:, seg])
plot(sisdr vs minutes)                  # look for the knee
```

### Calibration protocol (run once, offline)

```python
# on N long dev streams, log all three levels simultaneously
for stream in dev_streams:
    per-minute: sisdr[b], divergence[b], state_metrics[b]

# 1. verify correlation: scatter state_metric / divergence vs sisdr per bucket
#    keep only metrics with |spearman rho| > ~0.7 against sisdr
# 2. pick the alarm threshold at the knee:
#    thresh = value of metric at the first bucket where
#             sisdr drops > 0.5 dB below the 0-1 min bucket
# 3. sanity-check false-positive rate: fraction of buckets where
#    metric > thresh but sisdr is still healthy
```

Deployment usage after calibration:

- **first line:** Level 1 state metrics every frame (trigger for refresh/reset);
- **second opinion:** Level 2 divergence probe every ~60 s on real audio;
- **validation only:** Level 3 whenever the model or stabilization scheme changes.

Do NOT trust state norms alone without this calibration — the boundary between
"drifted but harmless" and "drifted and harmful" is model-specific and only
observable end-to-end.

---

## Phase 1 — Approach 1: training-free state adjustment

### 1.1 Adjuster hook

One class, three composable mechanisms. Applied after every `lstm_step`.

```python
class StateAdjuster:
    def __init__(self, ref, use_clip=True, gamma=None, gamma_dims=None,
                 restandardize_every=None, alpha=None):
        self.c_max = ref.c_p999            # per layer, per dim
        self.gamma = gamma                 # e.g. 0.9995, or None
        self.gamma_dims = gamma_dims       # runaway dims only, or None = all
        self.re_every = restandardize_every  # frames, e.g. 6000 (= 60 s)
        self.alpha = alpha                 # soft-pull factor, e.g. 0.999
        self.mu, self.sigma = ref.mu, ref.sigma
        self.run_mu = RunningStats()       # EMA of live state stats
        self.t = 0

    def __call__(self, states):
        new = []
        for l, (h, c) in enumerate(states):
            # (a) clip:   c <- clamp(c, -c_max, c_max)
            c = c.clamp(-self.c_max[l], self.c_max[l])

            # (b) leaky decay: c <- gamma * c   (on flagged dims or all)
            if self.gamma is not None:
                if self.gamma_dims is not None:
                    c[..., self.gamma_dims[l]] *= self.gamma
                else:
                    c = c * self.gamma

            # (c) periodic re-standardization toward reference stats
            self.run_mu.update(l, c)
            if self.re_every and self.t % self.re_every == 0 and self.t > 0:
                rm, rs = self.run_mu.stats(l)
                c = self.mu[l] + self.sigma[l] * (c - rm) / (rs + 1e-6)

            # (c') OR continuous soft pull:  c <- alpha*c + (1-alpha)*mu
            if self.alpha is not None:
                c = self.alpha * c + (1 - self.alpha) * self.mu[l]

            new.append((h, c))
        self.t += 1
        return new
```

Streaming loop becomes:

```python
out, states = model.lstm_step(feats, states)
tracker.update(states)          # keep tracking to verify the fix
states = adjuster(states)
```

### 1.2 Tuning protocol

Grid search on a long-audio dev set (>= 30 min items), plus a short-audio dev set
to verify no regression:

| knob | values to sweep |
|---|---|
| clip percentile | off, 99, 99.9 |
| gamma (all dims) | off, 0.9999, 0.9995, 0.999 |
| gamma (runaway dims only) | off, 0.999, 0.99 |
| re-standardize interval | off, 60 s, 300 s |

Metric: SI-SDR per 1-min bucket. Acceptance criteria:

- flat SI-SDR out to 30 min (bucket 25–30 min within 0.3 dB of bucket 0–1 min);
- <= 0.1 dB SI-SDR loss on the short-audio dev set;
- channel-binding consistency preserved (no permutation flips introduced by the
  adjuster — check with a permutation-flip counter against oracle references).

Start with clip + mild gamma only; add re-standardization only if needed.
If Phase 1 flattens the state norms but SI-SDR still sags -> Phase 2.

---

## Phase 2 — Approach 2: trainable state-refresh network

### 2.1 Data generation (frozen separator)

For each long mixture (>= 10 min, synthesized so drift is present):

```python
states = model.init_states()
ring = RingBuffer(seconds=W)                 # W = burn-in window, e.g. 4 s
for t, frame in enumerate(stream):
    ring.push(frame)
    out, states = model.lstm_step(feats(frame), states)
    if t in sample_times:                    # random times, t >= 5 min
        s_drift = clone(states)

        # fresh run: zero state over only the last W seconds
        s0 = model.init_states()
        y_fresh, s_fresh = model.run_chunk(ring.contents(), s0)
        y_drift = outputs of the drifted run over the same window (cache them)

        # permutation check over the burn-in window
        s_id = corr(y_fresh[0], y_drift[0]) + corr(y_fresh[1], y_drift[1])
        s_sw = corr(y_fresh[0], y_drift[1]) + corr(y_fresh[1], y_drift[0])
        if s_id < s_sw:
            continue    # DISCARD swapped pairs (see note below)

        # rollout target for distillation: next Delta seconds from s_fresh
        y_next_fresh = model.run_chunk(next_delta_frames, s_fresh)
        save(s_drift, s_fresh, next_delta_frames, y_next_fresh)
```

**Note on the permutation filter:** the channel binding is entangled in the state
vector, so a "swapped" fresh state cannot be fixed by permuting dimensions —
discard those pairs (~50%) instead. Generate 2x the data to compensate.

Target: ~50–100k pairs across varied speakers/SNRs/drift ages.

### 2.2 Refresh network

Per-layer residual MLP (defaults to identity, which is the safe no-op):

```python
class StateRefresh(nn.Module):
    def __init__(self, hidden, width=2 * hidden):
        self.net = nn.Sequential(
            nn.LayerNorm(2 * hidden),        # input = concat(h, c)
            nn.Linear(2 * hidden, width), nn.ReLU(),
            nn.Linear(width, 2 * hidden),
        )
        nn.init.zeros_(self.net[-1].weight)  # start as identity
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, c):
        d = self.net(cat([h, c]))
        dh, dc = split(d)
        return h + dh, c + dc
```

One instance per LSTM layer (or shared with a layer embedding).

### 2.3 Training

```python
for s_drift, s_fresh, ctx, y_next_fresh in loader:
    s_hat = refresh(s_drift)                          # all layers
    loss = mse(s_hat, s_fresh)
    if distill:                                       # frozen separator
        y_hat = frozen_model.run_chunk(ctx, s_hat)
        loss += lam * mse(y_hat, y_next_fresh)        # lam ~ 1.0, Delta ~ 1 s
    loss.backward(); opt.step()
```

Train MSE-only first as a baseline; add distillation if output quality after
refresh is not matching the fresh run.

### 2.4 Inference integration

```python
REFRESH_EVERY = 10 * 100        # every 10 s (frames)
for t, frame in enumerate(stream):
    out, states = model.lstm_step(feats, states)
    tracker.update(states)
    if t % REFRESH_EVERY == 0 and t > 0:
        states = [refresh[l](h, c) for l, (h, c) in enumerate(states)]
```

Optional trigger instead of fixed interval: refresh only when
`c_norm > ref.norm_p99` (drift-gated).

### 2.5 Acceptance criteria

- SI-SDR flat out to 30+ min (same buckets as Phase 1);
- no audible click/transient at refresh instants (inspect spectrograms around
  refresh times; if audible, cross-fade 20–50 ms of pre/post-refresh output);
- **binding preserved:** permutation-flip rate across refresh instants ~0%
  (measure against oracle references on the dev set).

---

## Phase 3 — Final evaluation matrix

Run all configs through one harness:

| config | short-audio SI-SDR | SI-SDR @ 0–1 min | @ 14–15 min | @ 29–30 min | perm-flip rate |
|---|---|---|---|---|---|
| baseline (no fix) | | | | | |
| Phase 1 best | | | | | |
| Phase 2 (+ Phase 1) | | | | | |

Ship the simplest config that meets the acceptance criteria. Long-term root-cause
fix (when the backbone is next retrained): long-sequence / state-carry-over
fine-tuning with cell-norm regularization or LayerNorm-LSTM, which should make
both approaches optional safety nets.
