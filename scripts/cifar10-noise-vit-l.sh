python main.py --dataset c10 --noise-type clean --model-name vit-l --size 256

# Symmetric noise
for noise_rate in 0.2 0.4 0.6 0.8
do
  python main.py --dataset c10 --noise-type sn --noise-rate ${noise_rate} --model-name vit-l --size 256
done
# Pairflip noise
for noise_rate in 0.2 0.45
do
  python main.py --dataset c10 --noise-type pairflip --noise-rate ${noise_rate} --model-name vit-l --size 256
done