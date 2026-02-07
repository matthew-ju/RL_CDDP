import torch
from hostage_gridworld import HostageGridworld
from train_ppo import PPOAgent

def debug_ppo():
    env = HostageGridworld(bomb_timer=5, render_mode="human")
    agent = PPOAgent(obs_dim=5, action_dim=4, hidden_dim=128)
    
    # Load model
    try:
        agent.load_state_dict(torch.load('ppo_high_stakes.pt'))
        print("Loaded ppo_high_stakes.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    obs, info = env.reset()
    print("\nInitial State:")
    env.render()
    
    done = False
    step = 0
    total_reward = 0
    
    print("\nStarting Episode:")
    while not done:
        action = agent.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        action_name = ['Up', 'Down', 'Left', 'Right'][action]
        print(f"\nStep {step}: Action {action_name}, Reward {reward:.2f}")
        print(f"Position: {info['agent_pos']}")
        env.render()
        
        if terminated or truncated:
            done = True
            if info['agent_pos'] == env.hostage_pos:
                print("\nSUCCESS: Rescued Hostage!")
            else:
                print("\nFAILURE: Bomb Exploded or Timeout.")

if __name__ == "__main__":
    debug_ppo()
