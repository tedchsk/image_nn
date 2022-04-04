# nvidia-driver https://cloud.google.com/compute/docs/gpus/install-drivers-gpu

curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# miniconda 
sudo apt-get update
sudo apt-get install bzip2 libxml2-dev wget

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Project setup
# Continue from here
make envload
miniconda activate python39
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

