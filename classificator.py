'''
Created by Rubens Lopes
This script is used to evaluate the system with the test files.
'''
import tensorflow as tf, sys, os
import numpy as np

test_path = 'teste Ex4'
label_path = 'test/retrained_labels.txt'
model_path = 'test/retrained_graphEx4.pb'

label_lines = [line.rstrip() for line
 in tf.gfile.GFile(label_path)]

with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
	
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	dir_names = os.listdir(test_path)
	result = {}
	roc_data = {}
	num_exemples = 0

	for dir_class in dir_names:
		result[dir_class] = {}
		roc_data[dir_class] = []
		for key_class in dir_names:
			result[dir_class][key_class] = 0 
	


	for dir_class in dir_names:

		class_name = dir_class
		data_loc = test_path + '/' + dir_class
		file_names = os.listdir(data_loc)
		test_size = len(file_names)
		num_exemples += len(file_names)

		for file_name in file_names:
			image_path = data_loc + '/' + file_name

			try:
				image_data = tf.gfile.FastGFile(image_path, 'rb').read()
				predictions = sess.run(softmax_tensor, \
			             {'DecodeJpeg/contents:0': image_data})
				top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
				prediction_class = label_lines[top_k[0]]
				result[class_name][prediction_class] += 1
				score = predictions[0][top_k[0]]
				roc_data[class_name].append(np.array([top_k[0], score]))

			# case the image is corrupted			
			except tf.errors.InvalidArgumentError:
				os.remove(image_path)
	
	keys = []
	values = []
	for [key, value] in result.items():
		values.append(np.array([v for [k, v] in value.items()]))
		keys.append(key)
		print key + str(values[-1])

	values = np.array(values)
	print values

	for i in range(len(values)):
		print keys[i]
		precision = 1.0*values[i][i]/values[i].sum()
		recall = 1.0*values[i][i]/values[:,i].sum()
		f1 = 2*(precision*recall/(precision + recall))
		print 'Precisao: ' + str(precision)
		print 'Recall: ' + str(recall)
		print 'F1: ' + str(f1)
		