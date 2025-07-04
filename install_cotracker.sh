cd src/libs/submodules/cotracker
mkdir -p checkpoints
cd checkpoints
# download the online (multi window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
# download the offline (single window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ../../../../..

# build point trajectory optimizer
cd src/libs/submodules/optimize
PYTHON_EXECUTABLE=~/ssd/custom_colmap/.venv/bin/python
mkdir -p build && cd build
cmake -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} ..
make -j
cd ../../../../..
