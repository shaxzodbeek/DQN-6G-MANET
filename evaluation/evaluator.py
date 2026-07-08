import numpy as np
import pandas as pd
from env.manet_env import MANETEnv
from baselines.routing import ShortestPathRouting
from agents.dqn_agent import DQNAgent

def evaluate_protocols(num_nodes=50, test_episodes=300):
    env = MANETEnv(num_nodes=num_nodes, init_energy=200.0)
    baseline = ShortestPathRouting()

    # Yuklangan aqlli agent (1 dona yaratib hammaga ishlatamiz)
    smart_agent = DQNAgent(state_dim=4, epsilon=0.0)
    smart_agent.load_model()

    results = {
        "F-DQN": {"delivered": 0, "energy_used": 0.0, "hops": []},
        "ShortestPath": {"delivered": 0, "energy_used": 0.0, "hops": []}
    }

    print("\n🔬 Qiyosiy test boshlandi (F-DQN vs ShortestPath)...")

    for ep in range(test_episodes):
        env.step_mobility()
        active_nodes = np.where(~env.dead_nodes)[0]
        if len(active_nodes) < 2: break
        source, dest = np.random.choice(active_nodes, 2, replace=False)

        # 1. Shortest Path Test
        current_sp = source
        visited_sp = set([current_sp])
        hops_sp = 0
        for _ in range(30):
            next_hop = baseline.get_next_hop(env.G, current_sp, dest)
            if next_hop is None or next_hop in visited_sp: break
            env.transmit_packet(current_sp, next_hop)
            results["ShortestPath"]["energy_used"] += (env.e_tx + env.e_rx)
            visited_sp.add(next_hop)
            hops_sp += 1
            if next_hop == dest:
                results["ShortestPath"]["delivered"] += 1
                results["ShortestPath"]["hops"].append(hops_sp)
                break
            current_sp = next_hop

        # 2. F-DQN Test (Inference Mode)
        current_dqn = source
        visited_dqn = set([current_dqn])
        hops_dqn = 0
        for _ in range(30):
            neighbors = env.get_neighbors(current_dqn)
            valid_n = [n for n in neighbors if not env.dead_nodes[n] and n not in visited_dqn]
            if not valid_n: break

            neighbor_feats = []
            for n in valid_n:
                d_n = np.linalg.norm(env.positions[n] - env.positions[dest])
                d_c = np.linalg.norm(env.positions[current_dqn] - env.positions[dest])
                neighbor_feats.append([env.energies[current_dqn], d_c, env.energies[n], d_n])

            d_c = np.linalg.norm(env.positions[current_dqn] - env.positions[dest])
            current_state = [env.energies[current_dqn], d_c, np.mean([f[2] for f in neighbor_feats]), np.mean([f[3] for f in neighbor_feats])]

            next_hop, _ = smart_agent.select_next_hop(current_state, neighbor_feats, valid_n)
            env.transmit_packet(current_dqn, next_hop)
            results["F-DQN"]["energy_used"] += (env.e_tx + env.e_rx)
            visited_dqn.add(next_hop)
            hops_dqn += 1
            if next_hop == dest:
                results["F-DQN"]["delivered"] += 1
                results["F-DQN"]["hops"].append(hops_dqn)
                break
            current_dqn = next_hop

    df = pd.DataFrame({
        "Protocol": ["Shortest Path (Dijkstra)", "F-DQN (Ours)"],
        "PDR (%)": [(results["ShortestPath"]["delivered"]/test_episodes)*100, (results["F-DQN"]["delivered"]/test_episodes)*100],
        "Energy Consumed (J)": [round(results["ShortestPath"]["energy_used"], 2), round(results["F-DQN"]["energy_used"], 2)],
        "Avg Hop Count": [np.mean(results["ShortestPath"]["hops"]) if results["ShortestPath"]["hops"] else 0,
                          np.mean(results["F-DQN"]["hops"]) if results["F-DQN"]["hops"] else 0]
    })

    df.to_csv("project/results/Table_2_Performance_Comparison.csv", index=False)
    return df
