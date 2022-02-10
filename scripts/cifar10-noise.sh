# Symmetric noise
for noise_rate in 0.2 0.4 0.6 0.8
do
  python main.py --dataset c10 --nosie-type sn --noise-rate ${noise_rate}
done
# Pairflip noise
for noise_rate in 0.2 0.45
do
  python main.py --dataset c10 --nosie-type pairflip --noise-rate ${noise_rate}
done