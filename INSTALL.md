conda update -n base -c defaults conda -y
conda create -n med python=3.7.4 -y

conda activate med
conda install -y pytorch torchvision cudatoolkit=11.1 -c pytorch-lts -c nvidia

git clone --branch tik https://github.com/xcvil/OFA.git

cd OFA
python -m pip install pip==21.2.4
pip install -r requirements.txt

# When install pytorch-lightning with Conda, please do check that pytorch version (1.8.X) does not change during installation.
conda install pytorch-lightning -c conda-forge