from gym.envs.registration import registry, register, make, spec
from itertools import product
# "lbforaging:Foraging-2s-8x8-2p-2f-coop-v1"
sizes = range(6, 7)
players = range(2, 3)
foods = range(4, 5)
coop = [True]
_grid_obs = [False]
for s, p, f, c, grid in product(sizes, players, foods, coop, _grid_obs):
    register(
        id="Foraging-{0}x{0}-{1}p-{2}f{3}{4}-v1".format(s, p, f, "-coop" if c else "", "-grid" if grid else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 500,
            "force_coop": c,
            "_grid_observation": grid,
        },
    )
# for s, p, f, c, grid in product(sizes, players, foods, coop, _grid_obs):
#     for sight in range(10, 11):
#         register(
#             id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}{5}-v1".format(s, p, f, "-coop" if c else "", "" if sight == s else f"-{sight}s", "-grid" if grid else ""),
#             entry_point="lbforaging.foraging:ForagingEnv",
#             kwargs={
#                 "players": p,
#                 "max_player_level": 3,
#                 "field_size": (s, s),
#                 "max_food": f,
#                 "sight": sight,
#                 "max_episode_steps": 500,
#                 "force_coop": c,
#                 "_grid_observation": grid,
#             },
#         )
