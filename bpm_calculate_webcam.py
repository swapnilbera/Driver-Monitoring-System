import numpy as np
import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time
from scipy.spatial import distance
from imutils import face_utils
import dlib
from pygame import mixer


levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0
imgPlot = None

mixer.init()
mixer.music.load("music.wav")
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

realWidth = 640
realHeight = 480
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15

# Initialize FaceDetector
faceDetector = FaceDetector()

# Initialize LivePlot for heart rate detection
plotY =LivePlot(realWidth,realHeight,[60,120],invert=True)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, realWidth)
cap.set(4, realHeight)

font = cv2.FONT_HERSHEY_SIMPLEX
bpmTextLocation = (videoWidth // 2, 40)

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 10   #15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))
flag = 0
i=0
while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame, bboxs = faceDetector.findFaces(frame, draw=False)
    frameDraw = frame.copy()
    ftime = time.time()
    ptime=0
    fps = 1 / (ftime - ptime)
    ptime = ftime

    # Drowsiness detection
    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        shape = predict(frame, dlib.rectangle(x1, y1, x1 + w1, y1 + h1))
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[42:48]
        rightEye = shape[36:42]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frameDraw, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frameDraw, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frameDraw, "****************ALERT! DROWSINESS!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frameDraw, "****************ALERT! DROWSINESS!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0

    # Heart rate detection
    if bboxs:
        # Extract relevant code for heart rate detection from your original heart rate detection code
        # ...
        x1, y1, w1, h1 = bboxs[0]['bbox']
        cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize
        outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
        frameDraw[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

        bpm_value = bpmBuffer.mean()
        imgPlot = plotY.update(float(bpm_value))

        # Display heart rate on the frame
        cvzone.putTextRect(frameDraw, f'BPM: {bpm_value}', bpmTextLocation, scale=2)
        imgPlot = plotY.update(float(bpm_value))

    # Display the processed frame
    imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)
    cv2.imshow("Combined Detection", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
