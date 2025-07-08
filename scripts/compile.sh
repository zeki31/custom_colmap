# cmake >= 3.28
sudo apt install build-essential checkinstall zlib1g-dev libssl-dev -y
wget https://github.com/Kitware/CMake/releases/download/v3.28.5/cmake-3.28.5.tar.gz
tar -zxvf cmake-3.28.5.tar.gz
cd cmake-3.28.5
sudo ./bootstrap
sudo make
sudo make install
hash -r
cmake --version
cd ..
sudo rm -rf cmake-3.28.5 cmake-3.28.5.tar.gz

# point trajectory optimizer
# set your customized python executable
# PYTHON_EXECUTABLE=/media/shaoliu/anaconda/envs/particlesfm/bin/python
PYTHON_EXECUTABLE=~/.venv/bin/python
cd src/matching/tracking/optimize
mkdir -p build && cd build
cmake -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} ..
make -j
cd ../../../../../

# GLOMAP
git clone git@github.com:colmap/glomap.git --recursive
cd glomap
mkdir build && cd build
cmake .. -GNinja
ninja
sudo ninja install
cd ../..
