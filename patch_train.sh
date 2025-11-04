# Fix the imports
sed -i 's/from custom_sampler import ExtendedSampler.*/import custom_sampler  # Monkey-patches Sampler/' train.py
sed -i 's/^# .*from gfn.samplers import Sampler.*/from gfn.samplers import Sampler/' train.py
sed -i 's/ExtendedSampler(/Sampler(/g' train.py

# Verify
grep -n "custom_sampler\|from gfn.samplers" train.py | head -5

# Run
python train.py --N 5 --num_steps 10
