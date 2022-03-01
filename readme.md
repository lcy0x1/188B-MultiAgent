# Installation

```
pip install PettingZoo
pip install ray[tune,rllib]
pip install dm-tree
pip install lz4
pip install tensorflow-gpu
pip install gputil
pip3 install torch torchvision torchaudio
pip install -e pz_vehicle
pip install -e gym_vehicle
pip install stable-baseline3
pip uninstall gym
pip install gym
```

```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install stable-baselines3
git clone https://github.com/lcy0x1/188B-MultiAgent.git
cd 188B-MultiAgent
pip install -e gym_vehicle
pip install -e gym_symmetric
```

python3 training/cls.py vehicle-v0 128 2 100 100 100 1
python3 training/cls.py vehicle-v0 128 3 100 100 100 1
python3 training/cls.py vehicle-v0 256 3 100 100 100 1
python3 training/cls.py vehicle-v0 256 4 100 100 100 1
python3 training/cls.py symmetric-v0 128 2 100 100 100 8
python3 training/cls.py symmetric-v0 128 3 100 100 100 8
python3 training/cls.py symmetric-v0 128 3 100 100 100 8
python3 training/cls.py symmetric-v0 128 4 100 100 100 8
