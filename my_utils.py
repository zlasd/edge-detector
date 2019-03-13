import tensorflow as tf

def printTensors(pb_file):
    """https://stackoverflow.com/questions/35336648/list-of-tensor-names-in-graph-in-tensorflow/50620593#50620593
    """
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)