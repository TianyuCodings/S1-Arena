#!/bin/bash
# Recreate the Python 3.12 environment for the BALROG-derived S1-Arena envs (game/open_world, game/rule_puzzles,
# game/roguelike/minihack_*, game/roguelike/nethack) on Linux x86_64, entirely in user space under $B.
# Box B layout: B=~/s1arena_env/balrog, repo synced to $B/S1-Arena. CLOCKPIN=1 builds the clock-pinning NLE variant.
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)  # the patch script sits next to this file
export B=${B:-$HOME/s1arena_env/balrog}
mkdir -p $B/bin $B/src $B/wheels $B/wheels_clockpin
export PATH=$B/bin:$PATH UV_CACHE_DIR=$B/.uv-cache UV_PYTHON_INSTALL_DIR=$B/.uv-python UV_PYTHON_BIN_DIR=$B/.uv-python-bin UV_NO_MODIFY_PATH=1

# 1. uv, and a uv-managed CPython 3.12 (it ships Python.h, which the system Pythons here lack)
[ -x $B/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$B/bin UV_NO_MODIFY_PATH=1 sh
uv python install 3.12
PY=$(uv python find --python-preference only-managed 3.12)

# 2. a C/C++ toolchain for NLE (no gcc, make, cmake, flex, bison on the box): conda-forge through micromamba
if [ ! -x $B/toolchain/bin/cmake ]; then
  mkdir -p $B/mamba && (cd $B/mamba && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba)
  MAMBA_ROOT_PREFIX=$B/mamba/root $B/mamba/bin/micromamba create -y -p $B/toolchain -c conda-forge --override-channels \
    cmake make flex bison gcc_linux-64=13 gxx_linux-64=13
fi

# 3. balrog-nle 0.9.0 (BALROG's NLE fork with the language wrapper) has wheels for CPython <= 3.11 only: build it for 3.12
mkdir -p $B/shims && printf '#!/bin/sh\nexit 128\n' > $B/shims/git && chmod +x $B/shims/git   # setup.py asks git for a commit
cd $B/src && [ -d balrog_nle-0.9.0 ] || curl -sL https://files.pythonhosted.org/packages/source/b/balrog-nle/balrog_nle-0.9.0.tar.gz | tar xz
SRC=$B/src/balrog_nle-0.9.0 WHEELS=$B/wheels
if [ "${CLOCKPIN:-0}" = 1 ]; then  # optional: NetHack reads its clock from NLE_FIXED_TIME when set (see nle_clockpin.patch.py)
  python3 $SCRIPT_DIR/nle_clockpin.patch.py $SRC $B/src/balrog_nle-0.9.0-clockpin && SRC=$B/src/balrog_nle-0.9.0-clockpin
  WHEELS=$B/wheels_clockpin
fi
T=$B/toolchain/bin
(cd $SRC && PATH=$T:$B/shims:$PATH CC=$T/x86_64-conda-linux-gnu-gcc CXX=$T/x86_64-conda-linux-gnu-g++ CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  MAKEFLAGS=-j8 uv build --wheel --python $PY -o $WHEELS .)

# 4. the venv
uv venv $B/venv --python $PY
PATH=$B/shims:$PATH uv pip install --python $B/venv/bin/python --find-links $WHEELS -e $B/S1-Arena \
  "numpy==1.26.4" "gym==0.23.0" "setuptools<81" "crafter==1.8.3" \
  "baba @ https://github.com/nacloos/baba-is-ai/archive/40ae1b29dbc67f2ecea06cc161ed5ca51719e5e3.tar.gz" \
  "balrog-nle==0.9.0" \
  "minihack @ https://github.com/balrog-ai/minihack/archive/3ecb6da4eadcac7a4dbc5f8e801ccf636e34d7ec.tar.gz" \
  "minigrid>=3.0" "gymnasium>=1.1"
# baba pulls opencv-python, which needs libGL; the headless build provides the same cv2 module
uv pip uninstall --python $B/venv/bin/python opencv-python
uv pip install --python $B/venv/bin/python --reinstall-package opencv-python-headless "opencv-python-headless" "numpy==1.26.4"
$B/venv/bin/python -c "import crafter, baba, nle, minihack, gym; from nle import nle_language_obsv; print('ok')"

# 5. play and score (results go to $B/S1-Arena/results/episodes/)
# cd $B/S1-Arena && export PATH=$B/venv/bin:$PATH
# s1arena play --model random --env game/open_world ; s1arena play --model random --env game/rule_puzzles --workers 12
# s1arena play --model random --env game/roguelike/minihack_mazewalk_9x9 ; s1arena play --model random --env game/roguelike/nethack
# s1arena env-score --models random --env game/open_world ; s1arena env-examples --env game/roguelike/nethack
