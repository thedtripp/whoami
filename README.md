# [~]# whoami_

### Who am I: a celebrity face match web application

## Purpose
This application aims to find images of celebrities that look similar to you based on the facial features in your face. Named for the linux command `whoami` which displays the current user's username.

## Examples

### Input: Elon Musk -> Output: Ben Affleck
<img src="https://user-images.githubusercontent.com/38776199/142755200-861eaa85-8088-405f-9ec4-f7ba47f9d899.jpg" width="400" height="200" />

### Input: "Hide the Pain" Harold -> Output: John Hawkes
<img src="https://user-images.githubusercontent.com/38776199/142755203-34761934-68bb-4780-9be4-9b4658d67631.jpg" width="400" height="200" />

### Input: Me -> Output: Kevin Hart
<img src="https://user-images.githubusercontent.com/38776199/142755094-4e0fbbe2-f3c2-47a8-a73c-c29c0d9ee6d5.png" width="400" height="200" />


## How it works
- Dataset
  - celebrity database consists of IMDB's top 1,000 actors and actresses
- Models
  - using ResNet50 and VGGFace pretrained networks

## How to use it:
- Only works on macOS for now
- Clone repo to your local machine
  - $ `git clone https://github.com/thedtripp/whoami.git`
- Change into the cloned repo
  - $ `cd whoami`
- Recommend using Anacona with a virtual environment running python 3.7.4
  - $ `virtualenv face-recognition --python=python3.7.4`
- Install required packages from requirements.txt file
  - $ `pip install -r requirements.txt`
- Start the application server
  - $ `python run.py`
- Open the application in web browser. Be sure to allow webcam access
  - `http://localhost:8050`
