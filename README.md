# Stable Diffusion Inference Optimization on A100

This repository contains the code and report for our High-Performance Machine Learning (HPML) course project:

> Stable Diffusion Inference Optimization on A100  
> Authors: Deeptanshu and Yashdeep Prasad

We optimize Stable Diffusion 1.5 inference on an NVIDIA A100 GPU, combining:

- Engineering: `torch.compile`, mixed precision, channels-last.
- Algorithmic: better schedulers and fewer denoising steps.
- Architectural: replacing the default VAE with a Tiny VAE.

Our goal is faster and lower-memory image generation without noticeable loss in visual quality.

---

## 1. Repository Structure

```text
.
├── HPML_Project.ipynb           # Main Colab notebook with all experiments
├── HPML_Project.pdf             # Short report / slide-style summary
└── README.md                    # This file
```

- `HPML_Project.ipynb` is the primary artifact. All baselines, optimizations, and measurements were run in this notebook on a Colab A100 GPU.
- `HPML_Project.pdf` contains presentation overview of the project and final results.

---

## 2. Environment and Setup

This project was authored primarily in a Google Colab environment on an NVIDIA A100 GPU.

### 2.1. Running in Google Colab (recommended)

1. Upload `HPML_Project.ipynb` to Colab or open it directly from GitHub in Colab.
2. Go to **Runtime → Change runtime type**:
   - Hardware accelerator: GPU
   - GPU type: A100 (if available)
3. Run the notebook cells from top to bottom.

The notebook:

- Installs all necessary Python packages with `pip` (PyTorch, Diffusers, Transformers, Accelerate, and others).
- Downloads Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`).
- Sets up the baseline pipeline and then applies each optimization step.

Note: We intentionally do not use TensorRT or xFormers. All optimizations are done with stock PyTorch and HuggingFace Diffusers.

### 2.2. Running Locally

To reproduce experiments locally on your own GPU:

1. Clone the repository:

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   # .venv\Scripts\activate     # Windows (PowerShell)
   ```

3. Install dependencies. Match what the notebook installs via `pip`. A typical setup will look like:

   ```bash
   pip install --upgrade pip

   # Core libraries
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Diffusers ecosystem
   pip install diffusers transformers accelerate safetensors

   # Optional utilities
   pip install tqdm pillow
   ```

   Adjust the CUDA version (`cu121`, `cu118`, etc.) based on your local driver and PyTorch compatibility.

4. Run the notebook locally:

   ```bash
   jupyter notebook HPML_Project.ipynb
   ```

   or convert parts of it into Python scripts if you prefer a pure-script workflow.

---

## 3. Baseline Setup

We start from the standard Stable Diffusion 1.5 pipeline:

- Model: `runwayml/stable-diffusion-v1-5`
- Resolution: 512 × 512
- Precision: `fp16`
- Scheduler: Euler Ancestral
- Steps: 30 denoising steps
- Hardware: NVIDIA A100 (GPU, Colab)
- Workload: single-image generation

Baseline performance:

| Config                                   | Time / image | Peak VRAM |
|------------------------------------------|--------------|-----------|
| Baseline SD 1.5 (Euler, 30 steps, fp16) | 1.30 s       | 2.8 GB    |

The UNet denoiser is the main hotspot for both time and memory.

---

## 4. Optimizations

We organize our optimizations into three categories:

1. Engineering: better use of PyTorch and GPU.
2. Algorithmic: better schedulers and fewer steps.
3. Architectural: swapping the VAE.

### 4.1. Engineering: `torch.compile` and GPU tuning

Pre-midterm work focused on classical GPU tuning:

- `torch.compile` on UNet and VAE:
  - Backend: `inductor`
  - Mode: `max-autotune` to aggressively optimize kernels.
- Channels-last memory format to improve convolution throughput.
- Enable TF32 matrix multiplies on A100 for faster matmul-heavy operations.

Result (compiled UNet and VAE):

| Config                       | Time / image | Peak VRAM |
|------------------------------|--------------|-----------|
| Baseline                     | 1.30 s       | 2.8 GB    |
| `torch.compile` (UNet + VAE) | 0.552 s      | 4.80 GB   |

This gives around a 2.3× speedup but about a 70 percent increase in VRAM compared to the baseline.

### 4.2. Architectural: Tiny VAE (TAESD)

We replace the default VAE decoder with AutoencoderTiny (TAESD), a distilled VAE that approximates the original decoder but is smaller and faster.

Gains:

- Faster decoding.
- Lower VRAM.

Result (compiled UNet + Tiny VAE):

| Config                        | Time / image | Peak VRAM |
|-------------------------------|--------------|-----------|
| Baseline                      | 1.30 s       | 2.8 GB    |
| Compiled UNet + full VAE      | 0.552 s      | 4.80 GB   |
| Compiled UNet + Tiny VAE      | 0.516 s      | 2.18 GB   |

The images are slightly softer in very fine details, but overall visual quality remains close to the original.

### 4.3. Algorithmic: Changing the denoising schedule

Denoising steps are expensive. Good schedulers can use fewer steps without sacrificing quality.

We explored:

- Schedulers:
  - Euler Ancestral.
  - DPMSolverMultistep (`dpmpp_2m`).
- Step counts:
  - 30 steps (baseline).
  - 20 steps (reduced).

Scheduler and step comparison:

| Config                      | Time / image | Peak VRAM |
|-----------------------------|--------------|-----------|
| Euler, 30 steps             | 0.547 s      | 5.13 GB   |
| Euler, 20 steps             | 0.395 s      | 5.11 GB   |
| DPM Solver, 20 steps        | 0.402 s      | 5.11 GB   |

Twenty-step schedules preserve quality while cutting compute, with Euler 20 steps giving the best raw latency in this batch of tests.

### 4.4. Algorithmic: Dynamic CFG cutoff

Classifier-Free Guidance (CFG) doubles UNet calls (conditional and unconditional) at every step. Later timesteps matter less for aligning with the prompt, so we can avoid the extra unconditional pass in late steps.

Implementation details:

- Use Diffusers’ `callback_on_step_end`.
- After a fraction of timesteps (for example 40 percent of total steps):
  - Drop the unconditional branch.
  - Set guidance scale effectively to 0 by editing:
    - `prompt_embeds`
    - `_guidance_scale`

This reduces effective UNet evaluations in later timesteps.

Dynamic CFG (no Tiny VAE yet):

| Config                                | Time / image | Peak VRAM |
|---------------------------------------|--------------|-----------|
| DPM Solver, 20 steps + CFG            | 0.347 s      | 5.11 GB   |
| Euler, 20 steps + CFG                 | 0.343 s      | 5.11 GB   |

---

## 5. Final Configuration and Results

The final system combines:

- Engineering:
  - `fp16`
  - Channels-last
  - `torch.compile` on UNet and VAE
- Algorithmic:
  - DPMSolverMultistep scheduler (`dpmpp_2m`)
  - 20 denoising steps
  - Dynamic CFG cutoff
- Architectural:
  - Tiny VAE for decoding (TAESD)

Final performance:

| Config                                            | Time / image | Peak VRAM |
|---------------------------------------------------|--------------|-----------|
| Baseline SD 1.5 (Euler, 30 steps, fp16)          | 1.30 s       | 2.8 GB    |
| Compiled UNet + VAE                              | 0.552 s      | 4.80 GB   |
| Compiled UNet + Tiny VAE                         | 0.516 s      | 2.18 GB   |
| DPM Solver (20 steps) + CFG                      | 0.347 s      | 5.11 GB   |
| Final: DPM Solver (20 steps) + CFG + Tiny VAE    | 0.297 s      | 4.13 GB   |

Key takeaways:

- The final pipeline is significantly faster than the baseline (over 4× speedup from 1.30 s to 0.297 s per image).
- Peak VRAM is still modest compared to some intermediate configurations and is acceptable on A100.
- Visual quality remains comparable to the baseline, with only minor differences in very fine details.

---

## 6. Reproducing the Experiments

To reproduce the full set of experiments:

1. Open `HPML_Project.ipynb`.
2. Run cells in order; each experiment section is clearly labelled:
   - Baseline timing.
   - `torch.compile` experiments.
   - Tiny VAE swap.
   - Scheduler variations.
   - Dynamic CFG implementation.
   - Final configuration.
3. For your own prompts:
   - Modify the `prompt` or `negative_prompt` variables.
   - Re-run the final configuration cell to generate new results.

For more stable measurements, run each configuration multiple times and average the timings. Use the same prompt and seed across configurations when visually comparing quality.

---
