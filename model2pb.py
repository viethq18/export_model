import os
import tensorflow as tf
from .model import Model

class GeneralModel():

    def __init__(self, specific_ckpt='', mapping_file=''):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._session = tf.Session(config=config)
        self._model = Model(
            specific_ckpt=specific_ckpt,
            mapping_file=mapping_file)

    def save(self, path=''):
        if not os.path.exists(path):
            os.makedirs(path)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self._model.sess,
            self._model.sess.graph.as_graph_def(),
            self._model.output_layer + self._model.output_attention_layer,
        )
        with tf.gfile.GFile(path + '/model.pb', "wb") as outfile:
            outfile.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    specific_ckpt = ''
    mapping_file = ''
    out_path = ''
    model = GeneralModel(specific_ckpt, mapping_file)
    model.save(out_path)