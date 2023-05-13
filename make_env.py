from main_to_env import AgentEnv
import env.scenarios as scenarios

def make_env(env_name,continuous_actions):

    # load scenario from script
    scenario = scenarios.load(env_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    scenario.reset_world(world)
    # create multiagent environment
    env = AgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,continuous_actions,scenario.done)

    env.act_dim = env.action_space.shape[0]   # 每一维代表该agent的act维度
    env.obs_dim = env.observation_space.shape[0]  # 每一维代表该agent的obs维度

    return env
