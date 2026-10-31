# Foul Falcon

### Computer-Vision Referee Assistance for FIRST Robotics Competition matches

Foul Falcon is a computer-vision system I developed to help human referees with rule violations during FRC matches.

During a match, six robots are moving quickly simultaneously, making it difficult for human referees to see every split-second violation. Foul Falcon takes a look at match footage to track robots, evaluate their interactions and positions, and flag violations.

## How It Works

The system uses a custom-trained YOLOv8 object-detection model, robot tracking, camera-perspective calibration, and rule-based foul detection.

## Development

I created 65,065 training annotations and trained, created, and analyzed seven detection models of...varying performance. I also developed the tracking, geometric contact-detection, auto field calibration, and game-rule logic.

## Technologies used

I used Python, Ultralytics YOLOv8, OpenCV, and Roboflow

## Project Background

I am the president and electrical lead of Vikings Robotics, FRC Team 2984. My experiences as both an FRC driver and volunteer referee motivated me to develop this project.
