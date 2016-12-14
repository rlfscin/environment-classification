'''
Created by Rubens Lopes
This script will count how many image of each class there are in the 'data_path' folder
'''
import tensorflow as tf, sys, os

data_path = 'test/counterData' 
label_path = 'test/retrained_labels.txt'
model_path = 'test/retrained_graphEx5.pb'

label_lines = [line.rstrip() for line
 in tf.gfile.GFile(label_path)]

with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
	
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	file_names = os.listdir(data_path)
	result = {}
	for label in label_lines:
		result[label] = 0
	result['other'] = 0

	for file_name in file_names:
		image_path = data_path + '/' + file_name
		try:
			image_data = tf.gfile.FastGFile(image_path, 'rb').read()
			predictions = sess.run(softmax_tensor, \
		             {'DecodeJpeg/contents:0': image_data})
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			prediction_class = label_lines[top_k[0]]
			if predictions[0][top_k[0]] > 0.95:
				result[prediction_class] += 1
			else:
				result['other'] += 1

		# case the image is corrupted			
		except tf.errors.InvalidArgumentError:
			os.remove(image_path)

	print 'count of each class'
	for [key, valeu] in result.items():
		print key + ' : ' + str(valeu)
