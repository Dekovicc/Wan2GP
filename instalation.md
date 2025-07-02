cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
>>enter>yes>enter>yes

source ~/.bashrc
cd /workspace
git clone https://github.com/Dekovicc/omnitalker_runpod
cd omnitalker_runpod

conda create -n wan2gp python=3.10.9
conda activate wan2gp

pip install -r requirements.txt

python wgp.py --i2v --multigpu --share