from gym.envs.registration import register

register(
    id='extractive-summary-v0',
    entry_point='gym_summarizer.envs:ExtractiveEnv',
)
# register(
#     id='abstractive-summary-v0',
#     entry_point='gym_foo.envs:AbstractiveEnv',
# )