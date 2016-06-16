import os, sys
import tensorflow as tf
import logging
import logging.config


LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': "[%(asctime)s] %(levelname)s " \
                "[%(threadName)s:%(lineno)s] %(message)s",
            'datefmt': "%Y-%m-%d %H:%M:%S"
        },
        'simple': {
            'format': '%(levelname)s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'verbose',
            'filename': 'rl.log',
            'maxBytes': 10*10**6,
            'backupCount': 3
            }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
    }
}


logging.config.dictConfig(LOGGING)

def getLogger(name):
    return logging.getLogger(name)


def restore_vars(saver, sess, checkpoint_dir):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.initialize_all_variables())

    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass

    path = tf.train.latest_checkpoint(checkpoint_dir)
    if path is None:
        print 'no existing checkpoint found'
        return False
    else:
        print 'restoring from %s' % path
        saver.restore(sess, path)
        return True
