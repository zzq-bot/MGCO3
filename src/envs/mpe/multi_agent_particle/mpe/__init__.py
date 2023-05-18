from gym.envs.registration import register
import mpe.scenarios as scenarios

def _register(scenario_name, gymkey):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )

scenario_name = "simple_spread"
gymkey = "SimpleSpread-v0"
_register(scenario_name, gymkey)