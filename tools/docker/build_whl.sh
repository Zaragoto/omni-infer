#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
set -exo pipefail

BASE_DIR=$(
    cd "$(dirname "$0")"
    pwd
)

# 代码下载需要网络代理
export http_proxy=${HTTP_PROXY}
export https_proxy=${HTTP_PROXY}
# The incoming value is a branch name (e.g. 'release_v0.6.0' or 'master').
# Use it directly if provided, otherwise default to 'master'.
branch="${BRANCH}"

# Determine OMNI_VERSION_NUM from the branch:
# - If branch is 'master', keep OMNI_VERSION_NUM='master'
# - If branch matches 'release_vMAJOR.MINOR(.PATCH)', parse into 'MAJOR.MINOR.PATCH'
#   missing patch defaults to 0 (so 'release_v0.6' -> '0.6.0')
if [ "${branch}" = "master" ]; then
    OMNI_VERSION_NUM="master"
else
    if [[ "${branch}" =~ ^release_v([0-9]+)\.([0-9]+)(\.([0-9]+))?$ ]]; then
        major="${BASH_REMATCH[1]}"
        minor="${BASH_REMATCH[2]}"
        patch="${BASH_REMATCH[4]:-0}"
        OMNI_VERSION_NUM="${major}.${minor}.${patch}"
    else
        echo "ERROR: could not parse version from branch '${branch}'.  Please provide a valid branch." 
        exit 1
    fi
fi

version_ge() {
    local ver1="$1"
    local ver2="$2"
    local -a arr1 arr2
    
    # 1. 拆分版本号为3段，空值补0（适配 0.5、1.0 等不完整格式）
    IFS='.' read -r -a arr1 <<< "${ver1:-0.0.0}"  # 若ver1为空，默认0.0.0
    IFS='.' read -r -a arr2 <<< "${ver2:-0.0.0}"  # 若ver2为空，默认0.0.0
    
    # 2. 逐段比较（主→次→修订），不足3段的补0
    for i in 0 1 2; do
        local num1=${arr1[$i]:-0}
        local num2=${arr2[$i]:-0}
        
        num1=$((num1))
        num2=$((num2))
        
        if [ "$num1" -gt "$num2" ]; then
            return 0
        elif [ "$num1" -lt "$num2" ]; then
            return 1
        fi
    done
    
    return 0
}

# 需要预先准备 omniinfer 代码，可根据CI流程优化，在容器中下载会导致构建时间过长
git config --global http.sslVerify false
# git clone -b "${branch}" https://gitee.com/omniai/omniinfer.git
# cd omniinfer/infer_engines
# git clone https://github.com/vllm-project/vllm.git

# Check if omniinfer exists in codes directory, copy if exists, otherwise clone
if [ -d "${BASE_DIR}/codes/omniinfer" ]; then
    echo "Directory ${BASE_DIR}/codes/omniinfer already exists. Copying to current path..."
    cp -r "${BASE_DIR}/codes/omniinfer" "${BASE_DIR}/"
    cd ${BASE_DIR}/omniinfer
    git checkout "${branch}" || echo "Warning: failed to checkout branch ${branch} in omniinfer."
else
    echo "Cloning omniinfer repo (branch: ${branch})..."
    if ! git clone -b "${branch}" https://gitee.com/omniai/omniinfer.git; then
        echo "ERROR: Failed to clone omniinfer from remote repository."
        echo "Please manually download omniinfer to ./codes/omniinfer and run this script again."
        exit 1
    fi
fi

cd ${BASE_DIR}/omniinfer/infer_engines

if [ -d "${BASE_DIR}/omniinfer/infer_engines/vllm" ]; then
    echo "vllm already exists in infer_engines. Skipping clone."
else
    # Check if vllm exists in codes directory, copy if exists, otherwise clone
    if [ -d "${BASE_DIR}/codes/vllm" ]; then
        echo "Directory ${BASE_DIR}/codes/vllm already exists. Copying to infer_engines..."
        cp -r "${BASE_DIR}/codes/vllm" "${BASE_DIR}/omniinfer/infer_engines/"
    else
        echo "Cloning vllm repo from remote..."
        if ! git clone https://github.com/vllm-project/vllm.git; then
            echo "ERROR: Failed to clone vllm from remote repository."
            echo "Please manually download vllm to ./codes/vllm and run this script again."
            exit 1
        fi
    fi
fi

# 构建 whl 包
cd ${BASE_DIR}/omniinfer
chmod +x build/build.sh
chmod +x infer_engines/bash_install_code.sh

if [ "${OMNI_VERSION_NUM}" = "master" ] || version_ge "${OMNI_VERSION_NUM}" "0.5.0"; then
    # For 'master' branch or versions >= 0.5.0 we use the cached build path
    bash -xe build/build.sh --cache 1
    cd ${BASE_DIR}/omniinfer/tools/quant/python
    python setup.py bdist_wheel
else
    bash -xe build/build.sh
fi

pip cache purge