from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.memoization import MemoizationPolicy
from RestaurentSearchBot.policy import RestaurantPolicy

logger = logging.getLogger(__name__)


def run_restaurent_bot_online(input_channel,interpreter):
    training_data_file = "/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/data/babi_task5_trn_rasa_with_slots.md"
    domain_path = '/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/restaurant_domain.yml'
    agent = Agent(domain=domain_path,policies=[MemoizationPolicy(), RestaurantPolicy()], interpreter=interpreter)
    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent


if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    model_path = "/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/models/nlu/current/default/model_20171013-223313"
    run_restaurent_bot_online(ConsoleInputChannel(), RasaNLUInterpreter(model_path))