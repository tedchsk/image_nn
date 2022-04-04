set -e # Stop the script if something fails


echo "Installing nvidia-driver"
# nvidia-driver https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

echo "Installilng miniconda"
# miniconda 
sudo apt-get update
sudo apt-get install bzip2 libxml2-dev wget

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc


echo "Setting up project"
# Continue from here
make envload
conda activate python39
# conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -e .

