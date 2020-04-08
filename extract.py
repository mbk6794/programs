import tensorflow as tf
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-n','--num',type=int,help='the number of layers')
parser.add_argument('-d','--dir',help='directory which has meta data')
args = parser.parse_args()

num = args.num
meta = glob.glob("{:s}/*.meta".format(args.dir))
meta.sort()
meta = meta[-1]

imported_graph = tf.train.import_meta_graph(meta)

w = []
for i in range(num):
    if i < num-1:
        w.append(tf.get_default_graph().get_tensor_by_name("hidden{:d}/kernel:0".format(i+1)))
    else:
        w.append(tf.get_default_graph().get_tensor_by_name("output/kernel:0"))

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

for i in range(num):
    w_star = sess.run(w[i])
    if i < num-1:
        np.savetxt("hidden{:d}.csv".format(i+1), w_star, delimiter=",")
    else:
        np.savetxt("output.csv", w_star, delimiter=",")
