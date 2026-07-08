import numpy as np
import torch
from env.manet_env import MANETEnv
from agents.dqn_agent import DQNAgent
from agents.federated_server import FederatedServer

def train_fdqn(num_nodes=50, episodes=2000, max_hops=30, fed_freq=10):
    env = MANETEnv(num_nodes=num_nodes, init_energy=200.0)
    server = FederatedServer()
    agents = [DQNAgent(state_dim=4) for _ in range(num_nodes)]

    rewards_history = []
    alive_nodes_history = []

    print(f"🚀 F-DQN O'qitish jarayoni boshlandi... ({episodes} epizod)")

    for ep in range(episodes):
        env.step_mobility()
        active_nodes = np.where(~env.dead_nodes)[0]
        if len(active_nodes) < 2: break

        source, dest = np.random.choice(active_nodes, 2, replace=False)
        current = source
        ep_reward = 0
        hops = 0
        visited = set([current])
        done = False

        while not done and hops < max_hops:
            neighbors = env.get_neighbors(current)
            valid_neighbors = [n for n in neighbors if not env.dead_nodes[n] and n not in visited]
            agent = agents[current]

            if not valid_neighbors:
                state_feat = [env.energies[current], np.linalg.norm(env.positions[current] - env.positions[dest]), 0, 0]
                agent.memory.push(state_feat, -10.0, None, True) # Jazo yumshatildi (-10)
                agent.train_step()
                ep_reward -= 10.0
                break

            neighbor_feats = []
            for n in valid_neighbors:
                d_n = np.linalg.norm(env.positions[n] - env.positions[dest])
                d_c = np.linalg.norm(env.positions[current] - env.positions[dest])
                neighbor_feats.append([env.energies[current], d_c, env.energies[n], d_n])

            d_c = np.linalg.norm(env.positions[current] - env.positions[dest])
            current_state = [env.energies[current], d_c, np.mean([f[2] for f in neighbor_feats]), np.mean([f[3] for f in neighbor_feats])]

            next_hop, _ = agent.select_next_hop(current_state, neighbor_feats, valid_neighbors)
            if not env.transmit_packet(current, next_hop): break

            visited.add(next_hop)
            hops += 1

            if next_hop == dest:
                reward = 100.0
                done = True
            else:
                reward = -1.0 # Har bir qadam uchun jazo

            next_states = []
            if not done:
                next_valid = [n for n in env.get_neighbors(next_hop) if not env.dead_nodes[n] and n not in visited]
                for nn in next_valid:
                    next_states.append([env.energies[next_hop], np.linalg.norm(env.positions[next_hop] - env.positions[dest]),
                                        env.energies[nn], np.linalg.norm(env.positions[nn] - env.positions[dest])])

            agent.memory.push(current_state, reward, next_states if not done else None, done)
            agent.train_step()
            current = next_hop
            ep_reward += reward

        for idx in active_nodes: agents[idx].decay_epsilon()

        rewards_history.append(ep_reward)
        alive_nodes_history.append(len(active_nodes))

        if (ep + 1) % fed_freq == 0:
            global_weights = server.aggregate_weights([agents[i].get_weights() for i in active_nodes])
            if global_weights:
                for i in active_nodes:
                    agents[i].set_weights(global_weights)
                    agents[i].update_target_network()

        if (ep + 1) % 100 == 0:
            print(f"Epizod: {ep+1:04d}/{episodes} | Mukofot(avg-50): {np.mean(rewards_history[-50:]):>6.2f} | Epsilon: {agents[active_nodes[0]].epsilon:.3f}")

    # O'qitish tugagach global modelni saqlaymiz
    if len(active_nodes) > 0:
        agents[active_nodes[0]].save_model()
        print("💾 Global AI Model xotiraga saqlandi.")

    return env, agents, rewards_history, alive_nodes_history
