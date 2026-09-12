#!/bin/bash

SI_ROOT=/home/${USER}/si

if [ ! -e "/home/${USER}/.ssh/id_ed25519" ]; then
    mkdir -p /home/${USER}/.ssh
    cp ${SI_ROOT}/keys/id_ed25519 /home/${USER}/.ssh/
    chmod 600 /home/${USER}/.ssh/id_ed25519
    cp ${SI_ROOT}/keys/id_ed25519.pub /home/${USER}/.ssh/
fi

cd $SI_ROOT

if [ ! -d "${SI_ROOT}/rcfiles" ]; then
    # TODO(madadam): Separate branch for this environment, or clean up rcfiles.
    git clone git@github.com:adamberenzweig/rcfiles.git 
fi
cd ${SI_ROOT}/rcfiles; make copy

if command -v conda >/dev/null 2>&1; then
    echo conda already installed
else
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
    # -b batch mode auto-accepts the license.
    sh Miniconda3-latest-Linux-x86_64.sh -b
    $HOME/miniconda3/bin/conda init bash
    # Reload to bring in conda.
    . $HOME/.bashrc
fi

if [ ! -d "${SI_ROOT}/chestnut-burr-detection-from-drone" ]; then
    git clone git@github.com:savannainstitute/chestnut-burr-detection-from-drone.git 
fi


if [[ -z $(conda env list | grep burr-detection) ]]; then
    # Use -y to auto-accept license prompts.
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    cd $SI_ROOT/chestnut-burr-detection-from-drone; conda env create -f burr-detection.yml -y

    conda run -n burr-detection pip install ipykernel
    conda run -n burr-detection python -m ipykernel install --user --name burr-detection --display-name burr-detection
fi

if [ ! -d "${SI_ROOT}/data/sample_data" ]; then
    mkdir -p $SI_ROOT/data; cd $SI_ROOT/data
    wget "https://drive.usercontent.google.com/download?id=1eUWmgBevc6CP5g-XBN4AgOzowgSOUxj3&export=download&confirm=t" -O sample_data.zip && unzip sample_data.zip
fi
