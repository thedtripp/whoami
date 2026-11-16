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

    def get_frame(self, capture=False):
        """Return (status, jpeg). status is 'live', 'no_face' or 'matched'.
        A match only runs when capture=True (the user pressed the button)."""
        ok, image = self.video.read()
        if not ok:
            return "no_camera", None
        clean = image.copy()
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x-p, y-p+2), (x+w+p, y+h+p+2), (0, 255, 0), 2)

        if capture:
            if len(faces) == 0:
                return "no_face", self._encode(cv2.flip(image, 1))
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
            face = clean[max(y-p+2, 0):y+h+p-2, max(x-p+2, 0):x+w+p-2]
            if face.size == 0:
                return "no_face", self._encode(cv2.flip(image, 1))
            return "matched", self._encode(cfm.main(face))

        return "live", self._encode(cv2.flip(image, 1))

    @staticmethod
    def _encode(img):
        _, frame = cv2.imencode('.jpg', img)
        return frame

def _data_url(frame):
    return f"data:image/jpeg;base64, {base64.b64encode(frame.tobytes()).decode()}"

# Setup small Quart server for streaming via websocket.
server = Quart(__name__)
delay_between_frames = 0.05  # add delay (in seconds) if CPU usage is too high
RESULT_HOLD_SECONDS = 10

@server.websocket("/stream")
async def stream():
    camera = VideoCamera(0)  # zero means webcam
    loop = asyncio.get_running_loop()
    capture_requested = False

    async def listen():
        nonlocal capture_requested
        while True:
            msg = await websocket.receive()
            if msg == "capture":
                capture_requested = True

    listener = asyncio.ensure_future(listen())
    try:
        while True:
            await asyncio.sleep(delay_between_frames)
            capture, capture_requested = capture_requested, False
            status, frame = await loop.run_in_executor(None, camera.get_frame, capture)

            if status == "no_camera":
                await websocket.send("status:Can't read the camera. Check browser/terminal camera permission.")
            elif status == "no_face":
                await websocket.send("status:No face found. Face the camera and try again.")
                await websocket.send(_data_url(frame))
            elif status == "matched":
                await websocket.send("status:Match found! Resuming live view shortly...")
                await websocket.send(_data_url(frame))
                await asyncio.sleep(RESULT_HOLD_SECONDS)
                await websocket.send("status:")
            else:
                await websocket.send(_data_url(frame))
    finally:
        listener.cancel()

def step(num, title, text):
    return html.Div(className="step", children=[
        html.Div(num, className="step-num"),
        html.Div([html.Div(title, className="step-title"), html.P(text, className="step-text")]),
    ])

layout = html.Div([
    html.Div(className="topbar", children=[
        html.Span("[~]# whoami_", className="brand"),
        html.Span("a celebrity look-alike finder", className="tagline"),
    ]),
    html.Div(className="main", children=[
        html.Div(className="panel", children=[
            html.Div([html.Span(className="dot"), "live camera"], className="panel-head"),
            html.Div(className="panel-body", children=[
                html.Img(id="video"),
                html.Div(className="capture-row", children=[
                    html.Button("● Capture", id="capture-btn", className="capture-btn", n_clicks=0),
                    html.Span(id="status", className="status"),
                ]),
            ]),
        ]),
        html.Div(className="panel", children=[
            html.Div("how it works", className="panel-head"),
            html.Div(className="panel-body", children=[
                step("1", "Get ready", "One person, face clearly visible, ideally front-on. Take your time."),
                step("2", "Press Capture", "When the green box is on your face, hit the button. Nothing is captured until you do."),
                step("3", "Meet your match", "The network compares you against IMDB's top 1,000 actors and shows the closest."),
            ]),
        ]),
    ]),
    html.Div("made by DDTripp", className="footer"),
    WebSocket(url="ws://127.0.0.1:5000/stream", id="ws"),
])

app.layout = layout

# Copy data from websocket to the Img element (images) or status text.
app.clientside_callback(
    """function(m){
        const nu = window.dash_clientside.no_update;
        if(!m) return [nu, nu];
        if(m.data.startsWith('status:')) return [nu, m.data.slice(7)];
        return [m.data, nu];
    }""",
    Output("video", "src"),
    Output("status", "children"),
    Input("ws", "message"),
)

# Tell the server to capture when the button is pressed.
app.clientside_callback(
    "function(n){ return n ? 'capture' : window.dash_clientside.no_update; }",
    Output("ws", "send"),
    Input("capture-btn", "n_clicks"),
)

if __name__ == '__main__':
    threading.Thread(target=app.run).start()
    server.run()