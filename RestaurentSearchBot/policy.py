from rasa_core.policies.keras_policy import KerasPolicy
import logging


class RestaurantPolicy(KerasPolicy):
    def _build_model(self, num_features, num_actions, max_history_len):
        """Build a keras model and return a compiled model.
        :param max_history_len: The maximum number of historical turns used to
                                decide on next action"""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        n_hidden = 64  # size of hidden layer in LSTM
        # Build Model
        batch_shape = (None, max_history_len, num_features)

        model = Sequential()
        model.add(Masking(-1, batch_input_shape=batch_shape))
        model.add(LSTM(n_hidden, batch_input_shape=batch_shape))
        model.add(Dense(input_dim=n_hidden, output_dim=num_actions))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logging.debug(model.summary())
        return model
