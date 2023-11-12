# Driver Drowsiness Detection System

This code can detect your eyes and alert when the user is drowsy and it monitors heart rate using camera.

## Applications 
This can be used by riders who tend to drive for a longer period of time that may lead to accidents

### Code Requirements 
The example code is in Python ([version 3.9.13])

### Install all the system requirments by:
pip install -r requirements.txt

### Description 

A computer vision system that can automatically detect driver drowsiness in a real-time video stream and then play an alarm if the driver appears to be drowsy.Here we are also monitoring heart rate in real-time using a webcam.This is based on an algorithm called Eulerian Video Magnification which makes it possible to see the colours of the face change as blood rushes in and out of the head. It is able to detect the pulses and calculates the heart rate in beats per minute (BPM).This method performs well in real-time, providing accurate results and maintaining a good frames-per-second rate, even when using a CPU.

### Algorithm for detecting drowsiness 

Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person), and then working clockwise around the eye.
It checks 20 consecutive frames and if the Eye Aspect ratio is less than 0.25, Alert is generated.

![eye1](https://github.com/swapnilbera/Driver-Monitoring-System/assets/87073046/e90e97a4-765f-4f6d-8802-a3599132fd32)

#### Relationship

![eye2](https://github.com/swapnilbera/Driver-Monitoring-System/assets/87073046/e84eeec5-3b91-4255-a548-792585fad939)

#### Summing up

![eye3](https://github.com/swapnilbera/Driver-Monitoring-System/assets/87073046/19830b0a-a263-4c66-9270-0f5d9f40a76d)

### STEPS FOR HEART RATE MONITORING

1.Input: Webcam video feed as the input for heart rate measurement.
   
2.Preprocessing: Use MediaPipe (CVZone) to detect and localize the face region in the video frames.

3.Spatial Decomposition: Decompose the video frames into multiple spatial frequency bands using a pyramid-based approach.

4.Temporal Filtering: Apply band-pass filtering techniques to isolate the desired frequency range associated with the heartbeat.

5.Magnification: Amplify the subtle temporal variations related to the heartbeat for better visibility.

6.Measurement: Extract the amplified signal and estimate the heart rate in beats per minute (bpm) using appropriate signal processing techniques.

7.Visualize Results: Use CVZone LivePlot to visualize the heart rate estimation results.

### Results 
![test result_eye detection](https://github.com/swapnilbera/Driver-Monitoring-System/assets/87073046/45c4b8e4-7a52-4c8b-bb41-52b2f99f9804)

![drowsiness_detection](https://github.com/swapnilbera/Driver-Monitoring-System/assets/87073046/1b025d05-a8df-4809-8887-a9336b469517)


## References 🔱
 
 -   Adrian Rosebrock, [PyImageSearch Blog](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
 -   https://people.csail.mit.edu/mrub/evm/
