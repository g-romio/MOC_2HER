# Multi-Updates Option Critic with Hindsight Experience Replay

#### Installation
```
virtualenv moc_cc --python=python3
source moc_cc/bin/activate
pip install tensorflow==1.15.0 
cd continuous_control
pip install -e . 
pip install gym==0.9.3
pip install mujoco-py==0.5.1
```

#### Launch
```
cd baselines/ppoc_int
python run_mujoco.py --env FetchPush-v2 --seed 35 --opt 2 --entcoeff 0.005 --optimsize 64 --kher 8 --hermvobj 1 --itschrew 150 --itsheroff 300 --objcoeff=0.8
```
