mkdir compile
cd compile

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

# COLMAP
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout 3.11.1
sudo apt-get install \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
# DO NOT run this command, it will cause ceres error
# sudo apt-get install -y \
#     nvidia-cuda-toolkit \
#     nvidia-cuda-toolkit-gcc
mkdir build
cd build
cmake .. -GNinja
ninja
sudo ninja install
cd ../..

# GLOMAP
git clone git@github.com:colmap/glomap.git --recursive
cd glomap
mkdir build && cd build
cmake .. -GNinja
ninja
sudo ninja install
cd ../..

# # TheiaSfM
# sudo apt-get install -y libopenimageio-dev librocksdb-dev rapidjson-dev freeglut3-dev
# cd src/submodules
# git clone https://github.com/B1ueber2y/TheiaSfM
# cd TheiaSfM
# git checkout upstream/particle-sfm
# mkdir build && cd build
# cmake ..
# make -j8
# sudo make install
# cd ../../../..

# # Point trajectory optimizer
# # set your customized python executable
# # PYTHON_EXECUTABLE=/media/shaoliu/anaconda/envs/particlesfm/bin/python
# PYTHON_EXECUTABLE=~/.venv/bin/python
# cd src/matching/tracking/optimize
# mkdir -p build && cd build
# cmake -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} ..
# make -j
# cd ../../../../../

# # gmapper (Particle-SFM)
# cd src/submodules/gmapper
# mkdir build && cd build
# cmake ..
# make -j
# sudo make install
# cd ../../../..

cd ..

# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd src/submodules/mast3r/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../../../
