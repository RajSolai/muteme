import pyautogui
import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

print("Mute Engine running !!")

# setting the initial state as "person in front of computer"
state = "person" 

# setting the screen size on your laptop
screenWidth, screenHeight = pyautogui.size()

# getting the mouse pointer
currentMouseX, currentMouseY = pyautogui.position()

# initialize webcam video object
cap = cv2.VideoCapture(0)

# width & height of webcam video in pixels -> adjust to your size
# adjust values if you see black bars on the sides of capture window
frameWidth = 1280
frameHeight = 720

# set width and height in pixels
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
# enable auto gain
cap.set(cv2.CAP_PROP_GAIN, 0)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# set state on state changes 
def set_state(given_state):
	global state
	if given_state!=state:
		state = given_state
		print("State changes to :"+state)
		zoom_audio_toogle()


def turn_on_audio(app):
	# the mute button changes for different
	# apps and for different screensizes
	coordinatemap = {
		"teams" : [],	
		"zoom" : zoom_audio_toogle,
	}
	pyautogui.click(1302, 13)

# toggle the zoom audio using keymap
def zoom_audio_toogle():
	# pyautogui.hotkey('alt', 'v') [currently the video does'nt working]
	pyautogui.hotkey('alt', 'a')


# main loop for the whole camera and predictions
# the model is trained using "teachablemachine.withgoogle.com"
while True:
	# Create the array of the right shape to feed into the keras model
	# The 'length' or number of images you can put into the array is
	# determined by the first position in the shape tuple, in this case 1.
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

	# Replace this with the path to your image
	check, frame = cap.read()
	#resize the image to a 224x224 with the same strategy as in TM2:
	#resizing the image to be at least 224x224 and then cropping from the center
	margin = int(((frameWidth-frameHeight)/2))
	square_frame = frame[0:frameHeight, margin:margin + frameHeight]
	# resize to 224x224 for use with TM model
	resized_img = cv2.resize(square_frame, (224, 224))
	# convert image color to go to model
	model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

	#turn the image into a numpy array
	image_array = np.asarray(model_img)

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

	# Load the image into the array
	data[0] = normalized_image_array

	# run the inference
	prediction = model.predict(data)
	if(prediction[0][0]>=0.8510176):
		set_state("no_person")
	else:
		set_state("person")
