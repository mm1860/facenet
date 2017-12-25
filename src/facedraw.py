from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import facenet
import align.detect_face
import cv2

with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
			
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
filename='C:/tf/facenet/datasets/who/5/1.jpg'
img = misc.imread(filename)
if img.ndim == 2:
	img = facenet.to_rgb(img)
img = img[:, :, 0:3]
total_boxes, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
# run detector
draw = cv2.imread(filename)
for b in total_boxes:
	cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

for indexrow,row in enumerate(points[:5]):
	for indexcol,col in enumerate(row):
		cv2.circle(draw, (col, points[indexrow+5][indexcol]), 1, (255, 255, 255), 2)

cv2.imshow("detection result", draw)
cv2.waitKey(0)
	
