import tensorflow as tf
import numpy as np

def build_model(height, width, n_frames, n_game_variables):
    # assume that the buttons are
    # available_buttons =
    # {
    #   # movements
    #   MOVE_FORWARD
    #   MOVE_BACKWARD
    #   MOVE_LEFT
    #   MOVE_RIGHT
    #   SPEED
    #   JUMP
    #
    #   # aim
    # 	ATTACK
    #   TURN_LEFT_RIGHT_DELTA
    #   LOOK_UP_DOWN_DELTA
    # }
    n_merge_filters = 32
    n_merge_output_channels = 16

    with tf.name_scope('model'):
        current_frames_ph = tf.placeholder('float', [None, height, width, 3 * n_frames], name='current_frames')
        game_variables_ph = tf.placeholder('float', [None, n_game_variables], name='game_variables')
        action_ph = tf.placeholder('float', [None, 9], name='true_action')
        keep_prob_ph = tf.placeholder('float', name='dropout')

        with tf.variable_scope('vision_model'):
            # conv1
            a1 = tf.contrib.layers.convolution2d(tf.nn.dropout(current_frames_ph / 255., keep_prob_ph), 32, (7, 7), tf.nn.relu, weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv1')

            # conv2
            a2 = tf.contrib.layers.convolution2d(tf.nn.dropout(a1, keep_prob_ph), 64, (7, 7), tf.nn.relu, stride=(2, 2), weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv2')

            # conv3
            a3 = tf.contrib.layers.convolution2d(tf.nn.dropout(a2, keep_prob_ph), 128, (17, 17), tf.nn.relu, stride=(2, 2), weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv3')

            # conv4
            vision_output = tf.contrib.layers.convolution2d(tf.nn.dropout(a3, keep_prob_ph), n_merge_filters, (9, 9), tf.nn.relu, stride=(2, 2), weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv4')

        with tf.variable_scope('game_variables_model'):
            gv1 = tf.contrib.layers.fully_connected(game_variables_ph, 32, activation_fn=tf.tanh,  weight_init=tf.contrib.layers.xavier_initializer(), name='fc1')

            game_variables_output = tf.contrib.layers.fully_connected(gv1, n_merge_filters * n_merge_output_channels, activation_fn=tf.tanh,  weight_init=tf.contrib.layers.xavier_initializer(), name='fc2')

        gv_filter = tf.reshape(game_variables_output, (1, 1, n_merge_filters, n_merge_output_channels), name='game_variables_model_filter')
        merged_stream = tf.nn.conv2d(vision_output, gv_filter, (1, 1, 1, 1), padding='SAME', name='merged_stream')

        with tf.variable_scope('tail'):
            t1 = tf.contrib.layers.convolution2d(tf.nn.dropout(merged_stream, keep_prob_ph), 32, (3, 3), tf.nn.relu, stride=(2, 2), weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv1')

            t2 = tf.contrib.layers.convolution2d(tf.nn.dropout(merged_stream, keep_prob_ph), 32, (3, 3), tf.nn.relu, stride=(2, 2), weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv2')

            # flatten
            t2_shape = t2.get_shape().as_list()
            flat_dim = np.product(t2_shape[1:])
            print 'final shape', t2_shape, 'flat_dim', flat_dim
            t2_flat = tf.reshape(t2, [-1, flat_dim])

            # binary buttons
            binary_buttons_logit = tf.contrib.layers.fully_connected(t2_flat, 7, weight_init=tf.contrib.layers.xavier_initializer(), name='binary_buttons_logit')

            # delta control
            delta_control = tf.contrib.layers.fully_connected(t2_flat, 2, weight_init=tf.contrib.layers.xavier_initializer(), name='delta_control')

        with tf.variable_scope('loss'):
            true_binary_buttons = tf.slice(action_ph, [0, 0], [-1, 7])
            true_delta_control = tf.slice(action_ph, [0, 7], [-1, -1])
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(binary_buttons_logit, true_binary_buttons), name='binary_buttons_loss') + tf.nn.l2_loss(true_delta_control - delta_control, name='delta_control_loss')

        # interfaces
        binary_buttons = tf.sigmoid(binary_buttons_logit, name='binary_buttons')
        pred_var = tf.concat(1, [binary_buttons, delta_control], name='predicted_action')
        # pred_var = tf.to_int32(tf.constant([0.1, 0.5, 0.9]*3))

    return loss, pred_var, (current_frames_ph, game_variables_ph, action_ph, keep_prob_ph)
