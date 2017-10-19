import logging

from examples.restaurant_example import RestaurantPolicy
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy

from RestaurentSearchBot.policy import RestaurantPolicy


def train_restaurent_search_dm():
    training_data_file= "/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/data/babi_task5_trn_rasa_with_slots.md"
    model_path = "/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/models/policy/current"
    domain_path = '/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/restaurant_domain.yml'
    agent = Agent(domain=domain_path,policies=[MemoizationPolicy(), RestaurantPolicy()])

    agent.train(
        training_data_file,
        max_history=3,
        epochs=100,
        batch_size=50,
        augmentation_factor=50,
        validation_split=0.2
    )

    agent.persist(model_path)


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    train_restaurent_search_dm()
