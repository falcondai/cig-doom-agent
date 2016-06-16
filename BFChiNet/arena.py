import os, sys, time

import numpy as np
from vizdoom import *
import tensorflow as tf

from model import *

from util import getLogger, restore_vars

# constants
width = 640
height = 480

if len(sys.argv) < 4:
    print 'python arena.py <checkpoint_dir> <n_frames> <n_game_variables>'
    sys.exit(1)
checkpoint_dir = sys.argv[1]
n_frames = int(sys.argv[2])
n_game_variables = int(sys.argv[3])
agent_config_path = 'config/limited.cfg'

# model
loss, prediction, (current_frames_ph, game_variables_ph, action_ph, keep_prob_ph) = build_model(height, width, n_frames, n_game_variables)

def sample_action(probs):
    buttons = []
    for p in probs:
        if np.random.rand() < p:
            buttons.append(1)
        else:
            buttons.append(0)
    return buttons

def main():
    game = DoomGame()
    game.add_game_args("+name BFChiNet +colorset %d" % np.random.randint(8))

    game.load_config(agent_config_path)

    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)
    with tf.Session() as sess:
        tf.set_random_seed(3)
        tf.initialize_all_variables().run()

        # restore session if there is a saved checkpoint
        restore_vars(saver, sess, checkpoint_dir)

        np.random.seed(1234)
        game.init()

        current_frames = np.zeros((1, height, width, 3 * n_frames))
        frame_counter = 0

        while not game.is_episode_finished():
            if game.is_player_dead():
                frame_counter = 0
                game.respawn_player()
                print
                print 'new life'

            state = game.get_state()
            if frame_counter < n_frames:
                # we dont have enough frames to feed the model
                current_frames[0,:,:,3*frame_counter:3*frame_counter+3] = state.image_buffer
                # just jump
                action = [0, 0, 0, 0, 0, 1, 0, 0, 0]
            else:
                # shift the frames
                current_frames[0,:,:,:-3] = current_frames[0,:,:,3:]
                current_frames[0,:,:,-3:] = state.image_buffer

                # we have enough frames to get a prediction
                pred = prediction.eval(feed_dict={
                    current_frames_ph: current_frames,
                    game_variables_ph: np.reshape(state.game_variables, (1, -1)),
                    keep_prob_ph: 1.,
                })[0]

                # sample binary buttons according to sigmoid
                action = sample_action(pred[:-2]) + np.asarray(pred[-2:], dtype='int32').tolist()

            game.make_action(action)
            print 'frame', frame_counter, 'variables', state.game_variables, 'action', action

            frame_counter += 1

    game.close()

if __name__ == '__main__':
    main()
