# Qwen3.6-A3B-8GB-llama.cpp

Run **Qwen3.6-35B-A3B UD-Q4_K_XL** with a 256K context on an 8 GB NVIDIA
GPU by keeping the MoE experts in system RAM.

Tested on an Intel Core Ultra 7 265HX, 64 GB RAM, and an 8 GB NVIDIA GPU
with llama.cpp and OpenCode/OpenChamber.

## Installation

Clone with the pinned llama.cpp submodule:

``` bash
git clone --recurse-submodules https://github.com/numsu/qwen3.6-a3b-8gb-llamacpp.git
cd qwen3.6-a3b-8gb-llamacpp
```

Apply the included hybrid-context checkpoint fix:

``` bash
./apply-patches.sh
```

Build llama.cpp with CUDA:

``` bash
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build llama.cpp/build --config Release -j
```

Download **Qwen3.6-35B-A3B UD-Q4_K_XL** into:

``` text
models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

Then start the server:

``` bash
./serve-qwen.sh
```

The OpenAI-compatible endpoint is:

``` text
http://127.0.0.1:8080/v1
```

Model files are intentionally not included in this repository.

## What this setup does

The goal is to make a capable 35B-A3B MoE model practical on an 8 GB GPU
without sacrificing a large context window.

The final setup uses:

-   Qwen3.6-35B-A3B UD-Q4_K_XL
-   256K context
-   Q8_0 K/V cache
-   GPU acceleration for the dense workload
-   CPU/system RAM for MoE experts
-   tuned CPU-MoE prompt batching
-   vanilla decoding rather than speculative decoding

See `serve-qwen.sh` for the exact runtime configuration.

## Performance

Representative results from real OpenCode workloads:

  Workload              Performance
  -------------- ------------------
  Cold prefill     \~875--935 tok/s
  Generation         \~35--38 tok/s
  Context            262,144 tokens
  KV cache              Q8_0 / Q8_0

Longer cold-prefill testing remained around 846 tok/s at roughly 58K
processed tokens.

These numbers are specific to the tested machine. CPU memory bandwidth
in particular matters a lot for this setup.

## Why base Qwen?

Ornith and Tiel were tested first and performed well, particularly as
coding-oriented models.

For the actual development workload, however, base Qwen3.6 proved more
reliable on project-specific language, terminology, and code context.
The specialized derivatives occasionally made small but important
text/code mistakes that base Qwen handled correctly.

That made base Qwen the better practical choice even when another model
or decoding strategy looked attractive on aggregate benchmarks.

The takeaway is simple: benchmark model quality on your own codebase,
not only on public coding leaderboards.

## Why no MTP?

MTP was tested with both Q4 and Q8 draft models.

It could be faster on favorable outputs, but performance varied
substantially with draft acceptance.

Representative results:

  Decoder                                    Result
  -------------------------------- ----------------
  Vanilla Qwen                       \~35--38 tok/s
  Q4 MTP-3, favorable run            \~38--40 tok/s
  Q4 MTP-3, lower-acceptance run       \~29.3 tok/s
  Q8 MTP-1                               \~34 tok/s
  Q8 MTP-2                               \~33 tok/s
  Q8 MTP-3                           \~25--26 tok/s

One Q4 MTP-3 workload reached only 52.8% draft acceptance and became
slower than vanilla decoding.

Speculative decoding also showed quality differences on some real
project inputs. The small and inconsistent speed advantage was not worth
the added variability.

## Why no DFlash?

DFlash was tested with:

``` text
n-max = 6
p-min = 0.85
```

Generation was only around:

``` text
15–17 tok/s
```

It was substantially slower than vanilla decoding and also showed
quality regressions on the real workload.

For this machine and model, DFlash was an easy rejection.

## Important tuning result

The biggest prefill improvement came from using CPU-MoE without
mmap-backed model access.

Before the final tuning, prompt processing was only a few hundred tokens
per second. After changing the load strategy and tuning batching, full
prompt chunks reached roughly 900+ tok/s.

The final batch/ubatch values are intentionally preserved in
`serve-qwen.sh`. Bigger values were not necessarily faster; there was a
sharp performance cliff beyond the tested sweet spot.

## llama.cpp patch

This repository includes one small llama.cpp patch for context
checkpoint selection on hybrid/recurrent models such as Qwen3.6.

Without it, changes to a long agent conversation could cause llama.cpp
to reject useful checkpoints and re-process a very large portion of the
prompt from scratch.

The patch was developed against llama.cpp commit:

``` text
a60f9aead0047b104807930b75fbc0a649f57779
```

It is kept as `hybrid-checkpoint.patch` rather than maintaining a
llama.cpp fork.

`apply-patches.sh` checks and applies it to the pinned submodule.

If you update llama.cpp, check whether upstream has fixed the underlying
checkpoint issue before attempting to reapply the patch.

## Repository layout

``` text
.
├── llama.cpp/                 # pinned upstream submodule
├── models/                    # GGUFs, ignored by Git
├── serve-qwen.sh              # tuned server configuration
├── hybrid-checkpoint.patch
├── apply-patches.sh
└── README.md
```

## OpenCode

Point an OpenAI-compatible OpenCode provider at:

``` text
http://127.0.0.1:8080/v1
```

Advertise the model with:

``` text
context: 262144
output: 32768
```

The exact OpenCode/OpenChamber configuration is deliberately left
separate from the inference configuration because plugins, reasoning
controls, and provider schemas change more frequently than llama.cpp
itself.

## Notes

This configuration assumes plenty of system RAM. The tested workstation
has 64 GB, and CPU-offloaded MoE weights are a core part of the design.

The setup is optimized for a single active inference slot. Parallel
requests work, but simultaneous CPU-MoE workloads compete heavily for
CPU and memory bandwidth.

If you change the model quant, llama.cpp revision, context size, or
speculative decoder, re-benchmark both performance and output quality on
your actual workload.

## License

This repository contains configuration, scripts, and a small llama.cpp
patch.

llama.cpp and the Qwen model weights remain subject to their respective
upstream licenses.
