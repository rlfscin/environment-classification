'''
Created by Rubens Lopes
This script will separate a raw data into folders of predicted classes
It will be used to filter raw data to help to label the data by the specialist
'''
import tensorflow as tf, sys, os

threshold = 0.75

data_path = 'test/allData' #path of the raw data
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
	for label in label_lines:
		os.mkdir(data_path + '/' + label)

	for file_name in file_names:
		image_path = data_path + '/' + file_name

		try:
			image_data = tf.gfile.FastGFile(image_path, 'rb').read()
			predictions = sess.run(softmax_tensor, \
		             {'DecodeJpeg/contents:0': image_data})
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			prediction_class = label_lines[top_k[0]]
			if predictions[0][top_k[0]] > threshold:
				os.rename(image_path, data_path + '/' + prediction_class + '/' + file_name)
				print file_name + ' was predicted as ' + prediction_class
			else:
				print file_name + " wasn't prediceted as anyclass" 

		# case the image is corrupted			
		except tf.errors.InvalidArgumentError:
			os.remove(image_path)
