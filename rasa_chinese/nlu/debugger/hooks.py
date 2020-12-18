import tensorflow as tf


class TensorObserveHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.output_flag = True

    def before_run(self, run_context):
        if self.output_flag:
            print(self.kwargs)
            self.output_flag = False
