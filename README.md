# Away detection

Welcome to Away detection , an simple script that **mutes** or even **turn off camera** in your favorite applications like 
- Microsoft teams ( in development) 
- Zoom Meeting ( completed implementation)
- Google Meeting ( in development)

### How it works

1. OpenCV reads your camera
2. The OpenCV loads camera input to Trained model
3. Pyautogui works by the output of the model output

### Requirements
```
- tensorflow
- opencv
- numpy
- pyautogui
``` 
### Installation
```
$ git clone https://github.com/RajSolai/muteme
$ cd muteme
$ python3 pydeskauto.py
```