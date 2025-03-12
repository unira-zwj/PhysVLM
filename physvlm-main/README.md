
## Get Started

### 1.Clone & Install

```shell
git clone git@github.com:unira-zwj/PhysVLM.git
cd PhysVLM/physvlm-main
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

---


### 2.Download the PhysVLM models to the checkpoints folder.

| Model                              | Links                                  |
|---------                           |---------------------------------------|
| PhysVLM-3B (Ready)                 | [`🤗HuggingFace`](...)    |
---


### 3.Inference

```shell
python start_physvlm_server.py
```

then you can request the server with `(app, host="0.0.0.0", port=8001)`, example: `inference.py` or `./eval/eval_phys_bench_sim.py`

---
