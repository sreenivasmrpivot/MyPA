# Download miniconda install script
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 
# Provide execute rights to the miniconda install script
# chmod +x Miniconda3-latest-Linux-x86_64.sh
# 
# Run the miniconda install script
# ./Miniconda3-latest-Linux-x86_64.sh 
# 
# Create a conda environment
conda create -n MyPA python=3.9

# Activate conda environment
conda activate MyPA

# Install required packages
pip install -r requirements.txt