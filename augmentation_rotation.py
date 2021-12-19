from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from matplotlib import pyplot
import os
import path


for img in os.listdir(path.path):
	# load the image
	image = load_img(path.path+str(img), target_size=(224, 224))
	# convert to numpy array
	data = img_to_array(image)
	# expand dimension to one sample
	samples = expand_dims(data, 0)
	# create image data augmentation generator
	datagen = ImageDataGenerator(rotation_range=90)
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)

	img_id = 0
	# generate samples and plot
	for _ in range(2):
		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		image = batch[0].astype('uint8')
		img_id += 1
		photo_path = path.path_augmented+f"{img}_rotation_"+str(img_id)+".jpg"
		cv2.imwrite(photo_path, image)
		print("Saved:", photo_path)
