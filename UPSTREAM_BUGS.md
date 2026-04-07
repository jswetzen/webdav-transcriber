# Upstream Library Bugs

These bugs were discovered during production testing (April 2026). They are patched
via files in `patches/` which are copied over the installed library files during the
Docker build. The `patches/` files are the definitive fixed versions.

---

## easyaligner — `data/collators.py`

### Bug 3: `vad_collate_fn` crashes with ragged audio batches

**Symptom:** `ValueError: could not broadcast input array from shape (N,) into shape (M,)`
when a batch contains audio clips of different lengths.

**Root cause:** `np.array(audios)` creates an object array (or raises) when clips have
different lengths; the code assumes a uniform tensor.

**Fix:** Pad all audio tensors to the maximum length in the batch before calling
`np.array()`.

**Upstream fix:** Add padding in `vad_collate_fn` before `np.array(audios)`.

---

### Bug 4: `vad_collate_fn` crashes when all batch items are `None`

**Symptom:** `AttributeError: 'NoneType' object has no attribute 'ndim'`

**Root cause:** No guard against `None` items; `max(a.shape[-1] ...)` fails on an empty
list after filtering.

**Fix:** Filter `None` items and items with `audio=None` before processing.

**Upstream fix:** Add `batch = [item for item in batch if item is not None and item.get("audio") is not None]` at the top of `vad_collate_fn`.

---

## easyaligner — `pipelines.py`

### Bug 7: `.half()` called before `.to(device)` breaks CPU inference

**Symptom:** `RuntimeError: "half" not implemented for 'Float'` (or similar) when running
on CPU, because `.half()` (float16) is only supported on CUDA.

**Root cause:** `features.half().to(device)` — `.half()` is applied unconditionally before
the device is known to be CUDA.

**Fix:** Remove `.half()`, use `features.to(device)` only.

**Upstream fix:** Replace `features.half().to(device)` with `features.to(device)` in
`emissions_pipeline_generator`.

---

## easytranscriber — `pipelines.py`

### Bug 6: `AutoModelForCTC` loaded with `.to("cuda").half()` hardcoded

**Symptom:** `RuntimeError: CUDA error` / model crashes immediately on CPU-only hosts.

**Root cause:** `AutoModelForCTC.from_pretrained(...).to("cuda").half()` — device and
dtype are hardcoded regardless of the `device` parameter.

**Fix:** Changed to `.to(device)` using the passed `device` argument.

**Upstream fix:** Replace `.to("cuda").half()` with `.to(device)` in the `pipeline`
function.

---

### Bug 11: `emissions_pipeline()` call missing `device=device`

**Symptom:** Emissions step always runs on CUDA even when `device="cpu"` is passed to
`pipeline()`.

**Root cause:** The `emissions_pipeline(...)` call inside `pipeline()` omits the
`device=device` argument, so it defaults to `"cuda"`.

**Fix:** Added `device=device` to the `emissions_pipeline(...)` call.

**Upstream fix:** Pass `device=device` explicitly in `easytranscriber/pipelines.py`.

---

### Bug 13: Default worker counts cause OOM on CPU hosts

**Symptom:** Container OOM-killed (RSS ~5 GB) on hosts with 8 GB RAM when processing
audio files longer than ~2 minutes.

**Root cause:** Defaults of `num_workers_features=4`, `num_workers_files=2` spawn too
many PyTorch DataLoader worker processes, each loading model weights into memory.

**Fix:** Reduced defaults to `num_workers_features=1`, `num_workers_files=1`,
`batch_size_features=4`. Peak RSS drops to ~3 GB.

**Upstream fix:** Lower defaults or document memory requirements; add a `device="cpu"`
fast-path that uses fewer workers.

---

## easytranscriber — `asr/ct2.py`

### Bug 8: `pin_memory=True` in DataLoader triggers CUDA initialisation

**Symptom:** `RuntimeError: Invalid device pointer` or CUDA initialisation failure on
CPU-only hosts, even though `device="cpu"` is set everywhere else.

**Root cause:** `pin_memory=True` (the PyTorch DataLoader default used here) silently
initialises the CUDA subsystem regardless of whether any model runs on GPU.

**Fix:** Set `pin_memory=False` in the `feature_dataloader` inside `transcribe()`.

**Upstream fix:** Set `pin_memory=False`, or gate it on `device == "cuda"`.
