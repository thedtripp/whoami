"""
file: run.py
Description: A Dash Application that captures image from web 
cam and runs face recognition program, returning a celebrity
resembling the user
"""
import asyncio
import base64
import cv2

import dash
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_extensions import WebSocket

import random
import threading
import time
from quart import Quart, websocket

import celeb_face_matcher as cfm

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = ">_ whoami"
server = app.server

gray=False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
p=35

class VideoCamera(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, image = self.video.read()
        #DECTECT FACE IN VIDEO CONTINUOUSLY   
        faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)#, Size(50,50))
        for (x, y, w, h) in faces_detected:
            image=cv2.rectangle(image, (x-p, y-p+2), (x+w+p, y+h+p+2), (0, 255, 0), 2)
            if random.random() > 0.9:
                image=image[y-p+2:y+h+p-2, x-p+2:x+w+p-2] #use only the detected face; crop it +2 to remove frame # CHECK IF IMAGE EMPTY (OUT OF IMAGE = EMPTY)     
                _, frame = cv2.imencode('.jpg', cfm.main(image))
                return True, frame

        image = cv2.flip(image,1)  
        _, frame = cv2.imencode('.jpg', image)
        return False, frame

# Setup small Quart server for streaming via websocket.
server = Quart(__name__)
delay_between_frames = 0.05  # add delay (in seconds) if CPU usage is too high

@server.websocket("/stream")
async def stream():
    camera = VideoCamera(0)  # zero means webcam
    while True:
        if delay_between_frames is not None:
            await asyncio.sleep(delay_between_frames)  # add delay if CPU usage is too high
        face, frame = camera.get_frame()

        if face:
            print("FACE DETECTED")
            await websocket.send(f"data:image/jpeg;base64, {base64.b64encode(frame.tobytes()).decode()}")
            wait_time = 10
            for _ in range(wait_time,0,-1):
                print(_)
                time.sleep(1)

        await websocket.send(f"data:image/jpeg;base64, {base64.b64encode(frame.tobytes()).decode()}")

def navbar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="#", target="_blank")),
        ],
        # brand="[~]$ whoami",
        brand="[~]# whoami_",
        color="#080808",
        sticky='top',
        dark=True,
        brand_style={'font-size': 48, 'color': '#25e62e'},
        style={'height': '80px', 'font-size': 24},
    )

def footer():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Made with ❤️ by DDTripp", href="#", target="_blank"), style={"font-size": "12px"}),
        ],
        color="#080808",
        sticky='bottom',
        dark=True,
        brand_style={'font-size': 48, 'color': '#25e62e'},
        style={'height': '50px', 'font-size': 24},
    )

# Create small Dash application for UI.
# app = dash.Dash(__name__)
layout = html.Div(children=[
    navbar(),
    html.Div(
        children=[
            dbc.Row(
                [
                    dbc.Col(html.Div(children=[
                        dbc.Card([
                            dbc.CardHeader("How to use it"),
                            dbc.CardBody([
                                html.H5("1. Take a webcam photo"),
                                html.P("There should be only one person in the photo. Recommendations: The face should be clearly visible, it’s better to use frontal photos."),
                                html.H5("2. The system detects the face"),
                                html.P("The system detects the face and creates a facial pattern. System facial point detection can locate the key components of faces, including eyebrows, eyes, nose, mouth and position. "),
                                html.H5("3. Enjoy the result"),
                                html.P("The Neural Network compares the person with celebrity faces and suggests the most similar one."),
                            ])
                        ])
                    ], style={"width": "100%", "box-shadow": "3px 3px 8px #c7c7c7", "border-radius": "2px"}), width=4),
                    dbc.Col(
                        html.Div(
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(html.Div(), width=2),
                                        dbc.Col(html.H1("Who Am I"), style={}),
                                        dbc.Col(html.H4("A Celebrity Look-Alike Application"), style={}),
                                        dbc.Col(html.Div(), width=2),
                                    ]),
                                    html.Img(
                                        id="video",
                                        style={"width":"100%", "display": "block", "margin": "auto"}
                                    )
                                ],
                                style={"width": "100%", "box-shadow": "3px 3px 8px #c7c7c7", "border-radius": "2px"}
                                )
                            ),
                        ), width=8,
                    ),
                ],
            ),
            dbc.Row([
                dbc.Col(html.Div()),
            ])
        ], style={"padding": "10px"}
    ),
    footer(),
    WebSocket(url=f"ws://127.0.0.1:5000/stream", id="ws")
])

app.layout = layout

# Copy data from websocket to Img element.
app.clientside_callback(
    "function(m){return m? m.data : '';}", 
    Output(f"video", "src"), 
    Input(f"ws", "message")
)

if __name__ == '__main__':
    threading.Thread(target=app.run_server).start()
    server.run()