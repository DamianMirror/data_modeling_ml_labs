from train import QLearningAgent
import gymnasium as gym

def test_agent(model_path="q_learning_car_racing.pkl", episodes=5):
    """Тестування навченої моделі"""
    env = gym.make("CarRacing-v3", render_mode="human", continuous=False)
    agent = QLearningAgent(action_space_size=5)
    agent.load(model_path)
    agent.epsilon = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=42)
        state = agent.discretize_state(obs)
        total_reward = 0
        print(f"Start Test Episode {episode + 1}")

        for _ in range(1000):
            action = agent.choose_action(state)
            obs, reward, term, trunc, _ = env.step(action)
            state = agent.discretize_state(obs)
            total_reward += reward
            env.render()
            if term or trunc: break

        print(f"Episode {episode + 1} Finished. Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test_agent()