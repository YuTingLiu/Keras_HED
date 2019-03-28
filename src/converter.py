import tensorflow as tf
import os
from tensorflow.python.framework import graph_io

from tensorflow.python.keras.models import load_model

from networks.hed import cross_entropy_balanced, ofuse_pixel_error
# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = '../checkpoints'
model_fname = '../checkpoints/HEDSeg/checkpoint.02-0.13.hdf5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

model = load_model(model_fname, custom_objects={'cross_entropy_balanced':cross_entropy_balanced,
                                                'ofuse_pixel_error': ofuse_pixel_error})

session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)
tf.train.write_graph(frozen_graph, save_pb_dir, 'hed.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, save_pb_dir, 'hed.pb', as_text=False)