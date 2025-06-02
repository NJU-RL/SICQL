import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import numpy as np
import common_args
import random
from dataset import ContextDataset
from model import RNNContextEncoder, RewardDecoder, StateDecoder
from utils import (
    build_meta_data_filename,
    build_context_model_filename,
)

from meta.src.envs import HalfCheetahVelEnv
from model import RNNContextEncoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)

    args = vars(parser.parse_args())
    print("Args: ", args)
    n_envs = args['n_envs']
    horizon = args['horizon']
    n_embd = args['n_embd']
    decoder_type = args['decoder_type']
    context_hidden_dim = args['context_hidden_dim']
    num_epochs = args['context_train_epochs']
    c_lr = args['context_lr']
    c_layer = args['c_layer']
    context_batch_size = args['context_batch_size']
    context_horizon = args['context_horizon']
    save_context_model_every = args['save_context_model_every']
    context_dim = args['context_dim']
    env = args['env']
    T = args['T']
    n_hists = args['hists']
    n_samples = args['samples']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_head = args['head']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    seed = args['seed']
    lin_d = args['lin_d']
    rollin_type = args['rollin_type']
    device = args['device']

    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    if not os.path.exists(f'figs/context/{rollin_type}/loss'):
        os.makedirs(f'figs/context/{rollin_type}/loss', exist_ok=True)
    if not os.path.exists(f'models/context/{rollin_type}'):
        os.makedirs(f'models/context/{rollin_type}', exist_ok=True)

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        "context_horizon": context_horizon,
        'dim': dim,
        'rollin_type': rollin_type,
    }

    model_config = {
        'shuffle': shuffle,
        'c_lr': c_lr,
        'context_dim': context_dim,
        'context_hidden_dim': context_hidden_dim,
        'c_layer': c_layer,
        'dropout': dropout,
        'n_head': n_head,
        'context_horizon': context_horizon,
        'dim': dim,
        'seed': seed,
    }

    if env.startswith('HalfCheetahVel'):
        with open(f'meta/datasets/{env}/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env_ = HalfCheetahVelEnv(tasks=tasks)
        max_action = float(env_.action_space.high[0])
        state_dim = env_.observation_space.shape[0]
        action_dim = env_.action_space.shape[0]
        dataset_config.update({'rollin_type': rollin_type})
        path_train = build_meta_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_meta_data_filename(
            env, n_envs, dataset_config, mode=1)
        filename = build_context_model_filename(env, model_config)
        loss_fn = torch.nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError

    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'c_layer': c_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
        'context_horizon':context_horizon,
        'device': device,
    }

    params = {
        'batch_size': context_batch_size,
        'shuffle': True,
    }

    log_filename = f'figs/context/{rollin_type}/loss/{filename}_logs.txt'
    with open(log_filename, 'w') as f:
        pass
    def printw(string):
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)
        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)


    train_dataset = ContextDataset(path_train, config)
    test_dataset = ContextDataset(path_test, config)

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    context_encoder = RNNContextEncoder(state_dim, action_dim, context_dim, context_hidden_dim, c_layer).to(device)
    state_decoder = StateDecoder(state_dim, action_dim, context_dim, context_hidden_dim).to(device)
    reward_decoder = RewardDecoder(state_dim, action_dim, context_dim, context_hidden_dim).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW([*context_encoder.parameters(), *reward_decoder.parameters(), *state_decoder.parameters()], lr=c_lr)

    test_loss = []
    train_loss = []
    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))

    best_loss = float('inf')
    for epoch in tqdm(range(num_epochs)):
        # EVALUATION
        printw(f"Epoch: {epoch + 1}")
        # start_time = time.time()
        with torch.no_grad():
            context_encoder.eval(); state_decoder.eval()
            # epoch_test_loss = 0.0
            transition, segment = next(iter(test_loader))
            state, action, reward, next_state, _, _= transition
            state_segment, action_segment, reward_segment = segment

            state = state.float().to(device)
            action =action.float().to(device)
            reward = reward.float().to(device)
            next_state = next_state.float().to(device)
            state_segment = state_segment.float().to(device)
            action_segment = action_segment.float().to(device)
            reward_segment = reward_segment.float().to(device)
            context = context_encoder(state_segment, action_segment, reward_segment)
            if decoder_type == 'reward':
                reward_predict = reward_decoder(state, action, next_state, context)
                loss = loss_fn(reward_predict, reward)
                print(
                    f'Predicted rewards: {reward_predict[:8].detach().cpu().numpy().reshape(-1)}')
                print(
                    f'   Real rewards  : {reward.detach()[:8].cpu().numpy()[:8].reshape(-1)}')
                # epoch_test_loss += loss.item()
            elif decoder_type == 'state':
                #预测差值
                next_state_predict = state_decoder(state, action, context)
                loss = F.mse_loss(next_state_predict, next_state-state)
                print(
                    f'Predicted state: {next_state_predict[:1].detach().cpu().numpy().reshape(-1)}')
                print(
                    f'   Real state  : {(next_state-state)[:1].detach().cpu().numpy().reshape(-1)}')
            else:
                reward_predict = reward_decoder(state, action, next_state, context)
                next_state_predict = state_decoder(state, action, context)
                loss = loss_fn(reward_predict, reward) + F.mse_loss(next_state_predict, next_state-state)
                print(
                    f'Predicted rewards: {reward_predict[:8].detach().cpu().numpy().reshape(-1)}')
                print(
                    f'   Real rewards  : {reward.detach()[:8].cpu().numpy()[:8].reshape(-1)}')
                print(
                    f'Predicted state: {next_state_predict[:1].detach().cpu().numpy().reshape(-1)}')
                print(
                    f'   Real state  : {(next_state-state)[:1].detach().cpu().numpy().reshape(-1)}')

            # test_loss.append(epoch_test_loss / len(test_loader))
            test_loss.append(loss.item())
            if test_loss[-1] < best_loss:
                best_loss = test_loss[-1]
                torch.save(context_encoder.state_dict(), f'models/context/{rollin_type}/{filename}_best.pt')
                print('Save the best model...')
        # TRAINING
        epoch_train_loss = 0.0
        context_encoder.train(); state_decoder.train()
        for i, (transition, segment) in enumerate(train_loader):
            state, action, reward, next_state, _, _= transition
            state_segment, action_segment, reward_segment = segment
            state = state.float().to(device)
            action =action.float().to(device)
            reward = reward.float().to(device)
            next_state = next_state.float().to(device)
            state_segment = state_segment.float().to(device)
            action_segment = action_segment.float().to(device)
            reward_segment = reward_segment.float().to(device)
            context = context_encoder(state_segment, action_segment, reward_segment)
            if decoder_type == 'reward':
                reward_predict = reward_decoder(state, action, next_state, context)
                loss = loss_fn(reward_predict, reward)
            elif decoder_type == 'state':
                next_state_predict = state_decoder(state, action, context)
                loss = F.mse_loss(next_state_predict, next_state-state)
            else:
                reward_predict = reward_decoder(state, action, next_state, context)
                next_state_predict = state_decoder(state, action, context)
                loss = loss_fn(reward_predict, reward) +  F.mse_loss(next_state_predict, next_state-state)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*context_encoder.parameters(), *reward_decoder.parameters(), *state_decoder.parameters()], 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        train_loss.append(epoch_train_loss / len(train_loader))
        # LOGGING
        if (epoch + 1) % save_context_model_every == 0:
            torch.save(context_encoder.state_dict(),
                       f'models/context/{rollin_type}/{filename}_epoch{epoch+1}.pt')
        # PLOTTING
        if (epoch + 1) % 10 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"Test Loss:        {test_loss[-1]}")
            printw(f"Train Loss:       {train_loss[-1]}")
            printw("\n")
            plt.yscale('log')
            plt.plot(train_loss[:], label="Train Loss")
            plt.plot(test_loss[:], label="Test Loss")
            plt.legend()
            plt.savefig(f"figs/context/{rollin_type}/loss/{filename}_train_loss.png")
            plt.clf()
    torch.save(context_encoder.state_dict(), f'models/context/{rollin_type}/{filename}.pt')
    print("Done.")
