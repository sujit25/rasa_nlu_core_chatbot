from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data
from rasa_nlu.model import Trainer

import logging
import six
if six.PY2:
    model_name = 'current_py2'
else:
    model_name = 'current_py3'


def train_restaurent_nlu():
    training_data = load_data('/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/data/franken_data.json')
    trainer = Trainer(RasaNLUConfig("/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/data/config_nlu.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('/home/sujit25/Workspace/RasaNlU_setup/RestaurentBot/RasaCoreAppDemo/models/nlu/current')
    return model_directory


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    train_restaurent_nlu()
