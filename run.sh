python3 context.py --env HalfCheetahVel-v0 --context_horizon 4 --context_lr 0.001 --context_hidden_dim 128 --context_dim 16 --context_batch_size 512  --context_train_epochs 200  --save_context_model_every 100 --horizon 200 --c_layer 1 --seed 1  --rollin_type 'mix' --decoder_type 'reward'  --device 'cuda:0'

python3 train.py  --env HalfCheetahVel-v0 --horizon 200 --context_horizon 4 --lr 3e-4 --n_layer 4 --m_layer 2 --head 1  --seed 1 --rollin_type 'mix' --num_epochs 400000  --freq 10000 --beta 1000.0 --n_embd 128  --context_hidden_dim 128 --context_dim 16  --c_layer 1  --context_epoch 200 --device 'cuda:0'

# Evaluate, choose an appropriate epoch
python3 eval.py --env HalfCheetahVel-v0 --horizon 200 --context_horizon 4 --lr 3e-4 --n_layer 4 --m_layer 2 --head 1  --seed 1  --rollin_type 'mix' --epoch 400000  --freq 10000 --beta 1000.0 --n_embd 128  --context_hidden_dim 128 --context_dim 16  --c_layer 1  --context_epoch 200 --device 'cuda:0'



