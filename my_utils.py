import tensorflow as tf
import numpy as np

def printTensors(pb_file):
    """ https://stackoverflow.com/questions/35336648/list-of-tensor-names-in-graph-in-tensorflow/50620593#50620593
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

        
        
def IoU(Reframe, GTframe):
    """ https://www.jianshu.com/p/a4237e252087
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
    """
    left1 = Reframe[1]
    top1 = Reframe[0]
    width1 = Reframe[3] - Reframe[1]
    height1 = Reframe[2] - Reframe[0]

    left2 = GTframe[1]
    top2 = GTframe[0]
    width2 = GTframe[3] - GTframe[1]
    height2 = GTframe[2] - GTframe[0]

    # endx = max(x1 + width1, x2 + width2)
    # startx = min(x1, x2)
    # width = width1 + width2 - (endx - startx)
    #
    # endy = max(y1 + height1, y2 + height2)
    # starty = min(y1, y2)
    # height = height1 + height2 - (endy - starty)

    #另一种求重叠区域面积的方式
    endx = min(left1+width1,left2+width2)
    startx = max(left1,left2)
    width = endx -startx

    endy = min(top1+height1,top2+height2)
    starty = max(top1,top2)
    height = endy - starty

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)

    return ratio



def var(nlist):
    N=len(nlist)
    narray = np.array(nlist)
    sum1 = narray.sum()
    narray2 = narray * narray
    sum2 = narray2.sum()
    mean = sum1 / N
    ans = sum2 / N - mean ** 2
    return ans