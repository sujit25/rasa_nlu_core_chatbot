from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter

nlu_model_path = '/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/models/nlu/current/default/model_20171013-223313'
policy_path = '/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/models/policy/current'


def run_restaurent_bot(server_forever=True):
    agent = Agent.load(policy_path, RasaNLUInterpreter(nlu_model_path))

    if server_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    run_restaurent_bot(server_forever=True)
