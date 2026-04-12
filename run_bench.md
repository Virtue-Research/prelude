# Micro-Benchmark

Two images:

| Image | Purpose | Content |
|-------|---------|---------|
| `prelude-microbench-dev` | Local dev | Toolchain + llama.cpp, mount source |
| `yuzhounie/prelude-microbench` | Remote / new device | Git clone + pull latest at startup |

## Local Dev

```bash
# Build dev image (once)
docker build -f benchmark/microbench/Dockerfile -t prelude-microbench-dev .

# Run (mount local source, incremental compile)
docker run --rm \
  -v $(pwd):/workspace \
  -v prelude-bench-target:/workspace/target \
  prelude-microbench-dev quant

# Edit kernel → re-run, no image rebuild needed
```

## Remote / New Device

```bash
# Build + push (after code update)
docker build -f benchmark/microbench/Dockerfile.release -t yuzhounie/prelude-microbench .
docker push yuzhounie/prelude-microbench

# Run anywhere (git pulls latest at startup)
docker run --rm yuzhounie/prelude-microbench quant
```

## Cross-Framework Comparison

Default workflow. Ours always runs fresh, baselines are cached.

```bash
# Local (uses run.sh, auto-builds dev image)
./benchmark/microbench/run.sh --all
./benchmark/microbench/run.sh --all quant
./benchmark/microbench/run.sh --all --force    # force re-run baselines

# Remote (clone repo first to get run.sh + baseline scripts)
git clone --recursive -b refactor https://github.com/Virtue-Research/prelude.git && cd prelude
./benchmark/microbench/run.sh --all
```

## Cross-Device Comparison

```bash
# Machine A
docker run --rm \
  -v $(pwd):/workspace \
  -v prelude-bench-target:/workspace/target \
  -v $(pwd)/results:/results \
  prelude-microbench-dev --json /results/machine_a.json

# Machine B
docker run --rm yuzhounie/prelude-microbench \
  --json /results/machine_b.json --compare /results/machine_a.json
```
