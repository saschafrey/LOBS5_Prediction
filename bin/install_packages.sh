
#pip install --upgrade jaxlib==0.4.2+cuda118
pip install --upgrade jaxlib==0.4.2+cuda116
pip install --upgrade "jax[cuda]==0.4.2" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install orbax==0.1.1
pip install einops
pip install tensorflow==2.11.*
pip install tensorRT==8.6.0
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.21.*
pip install flax==0.6.4
pip install pandas
pip install wandb
pip install tqdm
