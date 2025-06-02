import torch
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F

from ctrls.ctrl_meta import (
    MetaOptPolicy,
    MetaTransformerController,
)

from envs.meta_env import (
    MetaEnvVec,
)

from utils import convert_to_tensor


def deploy_online_vec(vec_env, controller, Heps, H, horizon, device):
    assert H % horizon == 0

    ctx_rollouts = H // horizon

    num_envs = vec_env.num_envs
    context_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.state_dim)).float().to(device)
    context_actions = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.action_dim)).float().to(device)
    context_next_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.state_dim)).float().to(device)
    context_rewards = torch.zeros(
        (num_envs, ctx_rollouts, horizon, 1)).float().to(device)

    cum_means = []
    for i in range(ctx_rollouts):
        batch = {
            'context_states': context_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
            'context_actions': context_actions[:, :i, :].reshape(num_envs, -1, vec_env.action_dim),
            'context_next_states': context_next_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
            'context_rewards': context_rewards[:, :i, :, :].reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
            controller)
        context_states[:, i, :, :] = convert_to_tensor(states_lnr, device=device)
        context_actions[:, i, :, :] = convert_to_tensor(actions_lnr, device=device)
        context_next_states[:, i, :, :] = convert_to_tensor(next_states_lnr, device=device)
        context_rewards[:, i, :, :] = convert_to_tensor(rewards_lnr[:, :, None], device=device)

        cum_means.append(np.sum(rewards_lnr, axis=-1))

    for _ in range(ctx_rollouts, Heps):
        # Reshape the batch as a singular length H = ctx_rollouts * horizon sequence.
        batch = {
            'context_states': context_states.reshape(num_envs, -1, vec_env.state_dim),
            'context_actions': context_actions.reshape(num_envs, -1, vec_env.action_dim),
            'context_next_states': context_next_states.reshape(num_envs, -1, vec_env.state_dim),
            'context_rewards': context_rewards.reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
            controller)

        mean = np.sum(rewards_lnr, axis=-1)
        cum_means.append(mean)

        # Convert to torch
        states_lnr = convert_to_tensor(states_lnr, device=device)
        actions_lnr = convert_to_tensor(actions_lnr, device=device)
        next_states_lnr = convert_to_tensor(next_states_lnr, device=device)
        rewards_lnr = convert_to_tensor(rewards_lnr[:, :, None], device=device)

        # Roll in new data by shifting the batch and appending the new data.
        context_states = torch.cat(
            (context_states[:, 1:, :, :], states_lnr[:, None, :, :]), dim=1)
        context_actions = torch.cat(
            (context_actions[:, 1:, :, :], actions_lnr[:, None, :, :]), dim=1)
        context_next_states = torch.cat(
            (context_next_states[:, 1:, :, :], next_states_lnr[:, None, :, :]), dim=1)
        context_rewards = torch.cat(
            (context_rewards[:, 1:, :, :], rewards_lnr[:, None, :, :]), dim=1)

    return np.stack(cum_means, axis=1)


def deploy_online_vec_c(context_encoder, context_horizon, vec_env, controller, Heps, H, horizon, device):
    assert H % horizon == 0

    ctx_rollouts = H // horizon

    num_envs = vec_env.num_envs
    context_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.state_dim)).float().to(device)
    context_actions = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.action_dim)).float().to(device)
    context_next_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.state_dim)).float().to(device)
    context_rewards = torch.zeros(
        (num_envs, ctx_rollouts, horizon, 1)).float().to(device)

    cum_means = []
    for i in range(ctx_rollouts):
        batch = {
            'context_states': context_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
            'context_actions': context_actions[:, :i, :].reshape(num_envs, -1, vec_env.action_dim),
            'context_next_states': context_next_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
            'context_rewards': context_rewards[:, :i, :, :].reshape(num_envs, -1, 1),
        }

        state_segment = F.pad(batch['context_states'], (0, 0, context_horizon, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
        action_segment = F.pad(batch['context_actions'], (0, 0, context_horizon, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
        reward_segment = F.pad(batch['context_rewards'], (0, 0, context_horizon, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
        context = context_encoder(state_segment.reshape(-1, state_segment.shape[-2], state_segment.shape[-1]), action_segment.reshape(-1, action_segment.shape[-2], action_segment.shape[-1]), reward_segment.reshape(-1, reward_segment.shape[-2], reward_segment.shape[-1]))
        batch['contexts'] = context.reshape(state_segment.shape[0], state_segment.shape[1], context.shape[-1])

        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
            controller)
        context_states[:, i, :, :] = convert_to_tensor(states_lnr, device=device)
        context_actions[:, i, :, :] = convert_to_tensor(actions_lnr, device=device)
        context_next_states[:, i, :, :] = convert_to_tensor(next_states_lnr, device=device)
        context_rewards[:, i, :, :] = convert_to_tensor(rewards_lnr[:, :, None], device=device)

        cum_means.append(np.sum(rewards_lnr, axis=-1))

    for _ in range(ctx_rollouts, Heps):
        # Reshape the batch as a singular length H = ctx_rollouts * horizon sequence.
        batch = {
            'context_states': context_states.reshape(num_envs, -1, vec_env.state_dim),
            'context_actions': context_actions.reshape(num_envs, -1, vec_env.action_dim),
            'context_next_states': context_next_states.reshape(num_envs, -1, vec_env.state_dim),
            'context_rewards': context_rewards.reshape(num_envs, -1, 1),
        }

        state_segment = F.pad(batch['context_states'], (0, 0, context_horizon-1, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
        action_segment = F.pad(batch['context_actions'], (0, 0, context_horizon-1, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
        reward_segment = F.pad(batch['context_rewards'], (0, 0, context_horizon-1, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
        context = context_encoder(state_segment.reshape(-1, state_segment.shape[-2], state_segment.shape[-1]), action_segment.reshape(-1, action_segment.shape[-2], action_segment.shape[-1]), reward_segment.reshape(-1, reward_segment.shape[-2], reward_segment.shape[-1]))
        batch['contexts'] = context.reshape(state_segment.shape[0], state_segment.shape[1], context.shape[-1])

        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
            controller)

        mean = np.sum(rewards_lnr, axis=-1)
        cum_means.append(mean)

        # Convert to torch
        states_lnr = convert_to_tensor(states_lnr, device=device)
        actions_lnr = convert_to_tensor(actions_lnr, device=device)
        next_states_lnr = convert_to_tensor(next_states_lnr, device=device)
        rewards_lnr = convert_to_tensor(rewards_lnr[:, :, None], device=device)

        # Roll in new data by shifting the batch and appending the new data.
        context_states = torch.cat(
            (context_states[:, 1:, :, :], states_lnr[:, None, :, :]), dim=1)
        context_actions = torch.cat(
            (context_actions[:, 1:, :, :], actions_lnr[:, None, :, :]), dim=1)
        context_next_states = torch.cat(
            (context_next_states[:, 1:, :, :], next_states_lnr[:, None, :, :]), dim=1)
        context_rewards = torch.cat(
            (context_rewards[:, 1:, :, :], rewards_lnr[:, None, :, :]), dim=1)

    return np.stack(cum_means, axis=1)


def online(envs, model, args, Heps, H, n_eval, state_dim, action_dim,  horizon, permuted=False, device=None):
    assert H % horizon == 0
    all_means_lnr = []
    lnr_controller = MetaTransformerController(
        model, batch_size=n_eval, device=device)
    vec_env = MetaEnvVec(envs, state_dim, action_dim)

    cum_means_lnr = deploy_online_vec(vec_env, lnr_controller, Heps, H, horizon, device)

    all_means_lnr = np.array(cum_means_lnr)
    means_lnr = np.mean(all_means_lnr, axis=0)
    sems_lnr = scipy.stats.sem(all_means_lnr, axis=0)

    # Plotting
    for i in range(n_eval):
        plt.plot(all_means_lnr[i], color='blue', alpha=0.2)

    plt.plot(means_lnr, label='Learner')
    plt.fill_between(np.arange(Heps), means_lnr - sems_lnr,
                     means_lnr + sems_lnr, alpha=0.2)
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title(f'Online Evaluation on {n_eval} Envs')

    return means_lnr, sems_lnr

def online_c(envs, model, context_encoder, context_horizon,  Heps, H, n_eval, state_dim, action_dim,  horizon, permuted=False, device=None):
    assert H % horizon == 0
    all_means_lnr = []
    lnr_controller = MetaTransformerController(
        model, batch_size=n_eval, device=device)
    vec_env = MetaEnvVec(envs, state_dim, action_dim)

    cum_means_lnr = deploy_online_vec_c(context_encoder, context_horizon, vec_env, lnr_controller, Heps, H, horizon, device)

    all_means_lnr = np.array(cum_means_lnr)
    means_lnr = np.mean(all_means_lnr, axis=0)
    sems_lnr = scipy.stats.sem(all_means_lnr, axis=0)

    # Plotting
    for i in range(n_eval):
        plt.plot(all_means_lnr[i], color='blue', alpha=0.2)

    plt.plot(means_lnr, label='Learner')
    plt.fill_between(np.arange(Heps), means_lnr - sems_lnr,
                     means_lnr + sems_lnr, alpha=0.2)
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title(f'Online Evaluation on {n_eval} Envs')

    return means_lnr, sems_lnr

def offline(envs, eval_trajs, model, args, n_eval, H, state_dim, action_dim,  permuted=False, device=None):
    all_rs_opt = []
    all_rs_lnr = []
    trajs = []

    for i_eval in range(n_eval):
        print(f"Eval env: {eval_trajs[i_eval]['goal']}")
        traj = eval_trajs[i_eval]
        with open(f"meta/datasets/{args['env']}/mix/dataset_task_{traj['goal']}.pkl", "rb") as f:
            datas = pickle.load(f)
        f.close()
        if args['env'] == 'Metaworld':
            returns = np.sort(datas['rewards'].reshape(-1,H).sum(axis=1))
        else:
            returns = np.sort(datas['rewards'].sum(axis=1))
        all_rs_opt.append(returns[-1])
        trajs.append(traj)

    print("Running meta offline evaluations in parallel")
    vec_env = MetaEnvVec(envs, state_dim, action_dim)
    lnr = MetaTransformerController(
        model, batch_size=n_eval, device=device)

    batch = {
        'context_states': convert_to_tensor([traj['context_states'] for traj in trajs], device=device),
        'context_actions': convert_to_tensor([traj['context_actions'] for traj in trajs], device=device),
        'context_next_states': convert_to_tensor([traj['context_next_states'] for traj in trajs], device=device),
        'context_rewards': convert_to_tensor([traj['context_rewards'][:, None] for traj in trajs], device=device),
    }

    lnr.set_batch(batch)
    _, _, _, rs_lnr = vec_env.deploy_eval(lnr)
    all_rs_lnr = np.sum(rs_lnr, axis=-1)

    baselines = {
        'Opt': np.array(all_rs_opt),
        'Learner': np.array(all_rs_lnr),
    }
    for k, v in baselines.items():
        print(k, np.mean(v))

    return baselines['Opt'], baselines['Learner']

def offline_c(envs, eval_trajs, model, context_encoder, context_horizon, args, n_eval, H, horizon,state_dim, action_dim,  permuted=False, device=None):
    all_rs_opt = []
    all_rs_lnr = []
    trajs = []
    for i_eval in range(n_eval):
        print(f"Eval env: {eval_trajs[i_eval]['goal']}")
        traj = eval_trajs[i_eval]
        with open(f"meta/datasets/{args['env']}/mix/dataset_task_{traj['goal']}.pkl", "rb") as f:
            datas = pickle.load(f)
        f.close()
        if args['env'] == 'Metaworld':
            returns = np.sort(datas['rewards'].reshape(-1,H).sum(axis=1))
        else:
            returns = np.sort(datas['rewards'].sum(axis=1))
        all_rs_opt.append(returns[-1])

        trajs.append(traj)

    print("Running meta offline evaluations in parallel")
    vec_env = MetaEnvVec(envs, state_dim, action_dim)
    lnr = MetaTransformerController(
        model, batch_size=n_eval, device=device)

    batch = {
        'context_states': convert_to_tensor([traj['context_states'] for traj in trajs], device=device),
        'context_actions': convert_to_tensor([traj['context_actions'] for traj in trajs], device=device),
        'context_next_states': convert_to_tensor([traj['context_next_states'] for traj in trajs], device=device),
        'context_rewards': convert_to_tensor([traj['context_rewards'][:, None] for traj in trajs], device=device),
    }

    state_segment = F.pad(batch['context_states'], (0, 0, context_horizon-1, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
    action_segment = F.pad(batch['context_actions'], (0, 0, context_horizon-1, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
    reward_segment = F.pad(batch['context_rewards'], (0, 0, context_horizon-1, 0, 0, 0)).unfold(1, context_horizon, 1)[:, :horizon, :, :].permute(0, 1, 3, 2)
    context = context_encoder(state_segment.reshape(-1, state_segment.shape[-2], state_segment.shape[-1]), action_segment.reshape(-1, action_segment.shape[-2], action_segment.shape[-1]), reward_segment.reshape(-1, reward_segment.shape[-2], reward_segment.shape[-1]))
    batch['contexts'] = context.reshape(state_segment.shape[0], state_segment.shape[1], context.shape[-1])

    lnr.set_batch(batch)
    _, _, _, rs_lnr = vec_env.deploy_eval(lnr)
    all_rs_lnr = np.sum(rs_lnr, axis=-1)

    baselines = {
        'Opt': np.array(all_rs_opt),
        'Learner': np.array(all_rs_lnr),
    }
    for k, v in baselines.items():
        print(k, np.mean(v))

    return baselines['Opt'], baselines['Learner']

