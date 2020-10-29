import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import config.Config as config
import os
import logging

########### Logger setup ##############
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########### Frozen model storage paths #####
save_pb_dir = config.model_pb_path
save_pb_name = config.model_pb_name
model_fname = config.model_path


# Clear any previous session.
tf.keras.backend.clear_session()

# Function to freeze a keras .h5 model to pb file
def freeze_graph(graph, session, output, save_pb_dir, save_pb_name, save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

######### Pb Folder create if not exists #############
try:
        os.mkdir("./models/pb")
except OSError:
        pass

model = load_model(model_fname)
session = tf.keras.backend.get_session()
INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
logger.info(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs],\
       save_pb_dir=save_pb_dir,save_pb_name=save_pb_name)