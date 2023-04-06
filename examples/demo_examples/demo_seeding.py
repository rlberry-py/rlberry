"""
 =====================
 Demo: demo_seeding
 =====================
"""
from rlberry.seeding import Seeder

seeder = Seeder(123)

# Each Seeder instance has a random number generator (rng)
# See https://numpy.org/doc/stable/reference/random/generator.html to check the
# methods available in rng.
seeder.rng.integers(5)
seeder.rng.normal()
print(type(seeder.rng))
# etc

# Environments and agents should be seeded using a single seeder,
# to ensure that their random number generators are independent.
from rlberry.envs import gym_make
from rlberry.agents import RSUCBVIAgent

env = gym_make("MountainCar-v0")
env.reseed(seeder)

agent = RSUCBVIAgent(env)
agent.reseed(seeder)

# Environments and Agents have their own seeder and rng.
# When writing your own agents and inheriring from the Agent class,
# you should use agent.rng whenever you need to generate random numbers;
# the same applies to your environments.
# This is necessary to ensure reproducibility.
print("env seeder: ", env.seeder)
print("random sample from env rng: ", env.rng.normal())
print("agent seeder: ", agent.seeder)
print("random sample from agent rng: ", agent.rng.normal())

# A seeder can spawn other seeders that are independent from it.
# This is useful to seed two different threads, using seeder1
# in the first thread, and seeder2 in the second thread.
seeder1, seeder2 = seeder.spawn(2)

# You can also use a seeder to seed external libraries (such as torch)
# using the function set_external_seed
from rlberry.seeding import set_external_seed

set_external_seed(seeder)
