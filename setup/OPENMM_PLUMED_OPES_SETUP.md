**Nothing is "wrong"** — it's just not organized well.

The problem is that the **environment variables** and **run commands** are inside the build section, which is confusing.

Here is the **clean, logical version** you should use:

```markdown
# OpenMM + PLUMED + OPES Setup Guide
**Last successful test:** July 20, 2026

This document contains everything you need to recreate the working OPES setup.

---

## 1. Build PLUMED with OPES (Run once)

```bash
cd /workspace/cse_md_sim/src
rm -rf plumed2
git clone https://github.com/plumed/plumed2.git
cd plumed2
./configure --enable-modules=opes --prefix=/workspace/cse_md_sim/.pixi/envs/default CXXFLAGS="-O3 -march=native -fPIC" --disable-mpi
make -j4
make install
echo "PLUMED with OPES built successfully!"
```

---

## 2. Set Environment Variables (Run every new session)

```bash
export PATH=/workspace/cse_md_sim/.pixi/envs/default/bin:$PATH
export PLUMED_KERNEL=/workspace/cse_md_sim/.pixi/envs/default/lib/libplumedKernel.so
export LD_LIBRARY_PATH=/workspace/cse_md_sim/.pixi/envs/default/lib:$LD_LIBRARY_PATH
```

---

## 3. Verify & Run Simulation

```bash
plumed --no-mpi info | grep -E "Version|OPES"

cd /workspace/cse_md_sim/src
python enhanced_sampling.py
python enhanced_analysis.py
```

---