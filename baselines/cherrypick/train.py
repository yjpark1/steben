
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

def train(args, env, agent, writer):
    start_time = time.time()
    global_step = 0
    history = {
        'reward': [],
        'solution': [],
        'cost': [],
    }
    
    cost_monitor = []
    cost_prev = np.inf
    num_fails = 0
    
    for episode in range(args.num_max_episodes):  # Run 100 episodes
        state, info = env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.replaybuffer.add(state, action, reward, next_state, terminated)
            loss, qvals = agent.learn(episode)
            if loss:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", qvals.mean().item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            state = next_state
            
            # logging
            global_step += 1
            total_reward += reward
            if terminated:
                history['reward'].append(total_reward)
                history['solution'].append(state.partial_solution)
                
                cost = env._get_current_cost()
                history['cost'].append(cost)
                
                cost_monitor.append(cost)
                writer.add_scalar("performance/reward", total_reward, episode)
                writer.add_scalar("performance/cost", cost, episode)
                writer.add_scalar("debug/epsilon", agent.epsilon, episode)
                break
            
        # save model
        if episode > 0 and episode % 100 == 0:
            cost_current = np.mean(cost_monitor)
            if global_step > args.learning_starts:
                if cost_current <= cost_prev:
                    torch.save(agent.network.state_dict(), f"{args.log_dir}/{args.filename_for_dqn_weights}.pth")
                    cost_prev = cost_current
                    num_fails = 0
                else: # fail to improve
                    num_fails += 1
                if num_fails > args.num_patience:
                    break
            
            cost_monitor = []
            mark = '*' if num_fails == 0 and global_step > args.learning_starts else ''
            msg = f"step: {global_step}, episode: {episode}, "
            msg += f"reward: {np.mean(history['reward'][-100:]): .3f}, "
            msg += f"cost: {cost_current: .3f} {mark}"
            print(msg)
    return history


