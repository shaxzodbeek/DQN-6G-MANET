import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from training.trainer import train_fdqn
from evaluation.evaluator import evaluate_protocols
import os

print("="*60)
print("F-DQN for MANET: Ilmiy Tajriba va Baholash Tizimi")
print("="*60)

# 1-Jadval: O'qitish va Simulyatsiya parametrlari (Maqola uchun)
params = {
    "Parameter": ["Number of Nodes", "Area Size", "Comm. Range", "Initial Energy", "Total Episodes", "Federated Freq"],
    "Value": ["50", "1000x1000 m", "250 m", "200 J", "2000", "10 episodes"]
}
df_params = pd.DataFrame(params)
df_params.to_csv("project/results/Table_1_Simulation_Params.csv", index=False)
print("\n📋 1-Jadval: Tajriba Parametrlari yaratildi.")

# O'QITISH JARAYONI
env, agents, rewards, alive_nodes = train_fdqn(num_nodes=50, episodes=2000, fed_freq=10)

# BAHOLASH JARAYONI
df_results = evaluate_protocols(num_nodes=50, test_episodes=300)
print("\n📊 2-Jadval: Protokollar Taqqoslamasi")
print(df_results.to_string(index=False))

# ILMIY GRAFIKLARNI CHIZISH
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Figure 1: Convergence
plt.figure(figsize=(8, 5))
smoothed_rewards = pd.Series(rewards).rolling(window=100).mean()
plt.plot(smoothed_rewards, color='#2ca02c', linewidth=2.5, label='F-DQN Moving Avg (100-ep)')
plt.title("Figure 1: Training Convergence of F-DQN over 2000 Episodes")
plt.xlabel("Episodes")
plt.ylabel("Accumulated Reward")
plt.legend()
plt.tight_layout()
plt.savefig("project/results/Figure_1_Convergence.png", dpi=300)
plt.close()

# Figure 2: Network Lifetime
plt.figure(figsize=(8, 5))
plt.plot(alive_nodes, color='#d62728', linewidth=2)
plt.title("Figure 2: Network Lifetime (Alive Nodes) Analysis")
plt.xlabel("Episodes")
plt.ylabel("Number of Active Nodes")
plt.fill_between(range(len(alive_nodes)), alive_nodes, color='#d62728', alpha=0.15)
plt.tight_layout()
plt.savefig("project/results/Figure_2_Network_Lifetime.png", dpi=300)
plt.close()

# Figure 3: QoS Performance Bar Charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PDR Subplot
sns.barplot(data=df_results, x="Protocol", y="PDR (%)", hue="Protocol", ax=axes[0], palette="Blues_d", legend=False)
axes[0].set_title("Packet Delivery Ratio (PDR)")

# Energy Subplot
sns.barplot(data=df_results, x="Protocol", y="Energy Consumed (J)", hue="Protocol", ax=axes[1], palette="Oranges_d", legend=False)
axes[1].set_title("Total Energy Consumption")

plt.suptitle("Figure 3: Comparative Analysis: F-DQN vs Shortest Path", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("project/results/Figure_3_Comparative_Analysis.png", dpi=300)
plt.close()

print("\n✅ Barcha grafiklar va jadvallar muvaffaqiyatli saqlandi: 'project/results/'")
