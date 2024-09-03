package com.opencv;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.highgui.HighGui;

import java.time.Duration;
import java.time.Instant;

public class LiveFaceDetector {
	private static final int EYE_BLINK_THRESHOLD = 3; // Number of consecutive eye blinks required for liveness
														// detection
	private static final long BLINK_TIME_THRESHOLD_MS = 300; // Minimum time difference between eye blinks (in
																// milliseconds)
	private static final long FACE_TIMEOUT_MS = 2000; // Timeout period for face detection (in milliseconds)

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// Load the trained haarcascade classifier XML files for face and eye detection
		CascadeClassifier faceCascade = new CascadeClassifier();
		faceCascade.load(
				"data/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_alt2.xml");

		CascadeClassifier eyeCascade = new CascadeClassifier();
		eyeCascade.load("data/raw.githubusercontent.com_anaustinbeing_haar-cascade-files_master_haarcascade_eye.xml");

		// Create a VideoCapture object to capture frames from the camera
		VideoCapture videoCapture = new VideoCapture(0); // 0 represents the default camera

		if (!videoCapture.isOpened()) {
			System.out.println("Failed to open the camera.");
			return;
		}

		// Set camera frame properties
		videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
		videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);

		// Create a window to display the camera feed
		HighGui.namedWindow("Live Face Detection");

		// Variables for eye blinking detection
		Instant lastBlinkTime = Instant.now();
		boolean isLive = false;

		// Continuously read frames from the camera feed
		Mat frame = new Mat();
		while (true) {
			// Read a frame from the camera
			videoCapture.read(frame);

			// Perform face detection on the frame
			MatOfRect faces = new MatOfRect();
			detectFaces(frame, faceCascade, faces);

			// Check if any faces are detected
			if (faces.toArray().length >= 1) {
				Rect faceRect = faces.toArray()[0];

				// Extract the region of interest (ROI) containing the face
				Mat faceROI = frame.submat(faceRect);

				// Detect eyes in the face ROI
				MatOfRect eyes = new MatOfRect();
				detectEyes(faceROI, eyeCascade, eyes);

				// Check if eyes are detected and determine if the person is live based on eye
				// blinking
				if (eyes.toArray().length >= 1) {
					Instant currentBlinkTime = Instant.now();
					Duration timeDifference = Duration.between(lastBlinkTime, currentBlinkTime);
					if (timeDifference.toMillis() <= BLINK_TIME_THRESHOLD_MS) {
						isLive = true;
						lastBlinkTime = currentBlinkTime;
					}
				} else {
					isLive = false;
				}

				// Draw rectangles around the face and eyes
				Imgproc.rectangle(frame, faceRect.tl(), faceRect.br(), new Scalar(0, 0, 255), 2);
				for (Rect eyeRect : eyes.toArray()) {
					Rect absoluteEyeRect = new Rect(faceRect.x + eyeRect.x, faceRect.y + eyeRect.y, eyeRect.width,
							eyeRect.height);
					Imgproc.rectangle(frame, absoluteEyeRect.tl(), absoluteEyeRect.br(), new Scalar(0, 255, 0), 2);
				}
			} else {
				// No face detected, consider it a spoof
				isLive = false;
			}

			// Display the frame with detected faces and eyes in the window
			HighGui.imshow("Live Face Detection", frame);

			// Check liveness status and display notification
			if (isLive) {
				System.out.println("Liveness: Real");
			} else {
				System.out.println("Liveness: Spoof");
			}

			// Exit the loop if the 'Esc' key is pressed
			if (HighGui.waitKey(1) == 27)
				break;
		}

		// Release the VideoCapture and close the window
		videoCapture.release();
		HighGui.destroyAllWindows();
	}

	private static void detectFaces(Mat frame, CascadeClassifier faceCascade, MatOfRect faces) {
		Mat grayFrame = new Mat();
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(grayFrame, grayFrame);
		faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
				new Size());
	}

	private static void detectEyes(Mat faceROI, CascadeClassifier eyeCascade, MatOfRect eyes) {
		Mat grayFaceROI = new Mat();
		Imgproc.cvtColor(faceROI, grayFaceROI, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(grayFaceROI, grayFaceROI);
		eyeCascade.detectMultiScale(grayFaceROI, eyes, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
				new Size());
	}
}

//
//import org.opencv.core.Core;
//import org.opencv.core.Mat;
//import org.opencv.core.MatOfRect;
//import org.opencv.core.Rect;
//import org.opencv.core.Scalar;
//import org.opencv.core.Size;
//import org.opencv.imgproc.Imgproc;
//import org.opencv.objdetect.CascadeClassifier;
//import org.opencv.objdetect.Objdetect;
//import org.opencv.videoio.VideoCapture;
//import org.opencv.videoio.Videoio;
//import org.opencv.highgui.HighGui;
//
//import java.time.Duration;
//import java.time.Instant;
//import java.util.ArrayList;
//import java.util.List;
//
//public class LiveFaceDetector {
//    private static final int EYE_BLINK_THRESHOLD = 3; // Number of consecutive eye blinks required for liveness detection
//    private static final long BLINK_TIME_THRESHOLD_MS = 500; // Minimum time difference between eye blinks (in milliseconds)
//
//    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
//        // Load the trained haarcascade classifier XML files for face and eye detection
//        CascadeClassifier faceCascade = new CascadeClassifier();
//        faceCascade.load("data/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_alt2.xml");
//
//        CascadeClassifier eyeCascade = new CascadeClassifier();
//        eyeCascade.load("data/raw.githubusercontent.com_anaustinbeing_haar-cascade-files_master_haarcascade_eye.xml");
//
//        // Create a VideoCapture object to capture frames from the camera
//        VideoCapture videoCapture = new VideoCapture(0); // 0 represents the default camera
//
//        if (!videoCapture.isOpened()) {
//            System.out.println("Failed to open the camera.");
//            return;
//        }
//
//        // Set camera frame properties
//        videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
//        videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
//
//        // Create a window to display the camera feed
//        HighGui.namedWindow("Live Face Detection");
//
//        // Variables for eye blinking detection
//        List<Instant> blinkTimestamps = new ArrayList<>();
//        boolean isLive = false;
//
//        // Continuously read frames from the camera feed
//        Mat frame = new Mat();
//        while (true) {
//            // Read a frame from the camera
//            videoCapture.read(frame);
//
//            // Perform face detection on the frame
//            MatOfRect faces = new MatOfRect();
//            detectFaces(frame, faceCascade, faces);
//
//            // For each detected face, perform eye detection and check for blinking
//            Rect[] facesArray = faces.toArray();
//            for (Rect faceRect : facesArray) {
//                // Extract the region of interest (ROI) containing the face
//                Mat faceROI = frame.submat(faceRect);
//
//                // Detect eyes in the face ROI
//                MatOfRect eyes = new MatOfRect();
//                detectEyes(faceROI, eyeCascade, eyes);
//
//                // Check if eyes are detected and determine if the person is live based on eye blinking
//                if (eyes.toArray().length >= 1) {
//                    blinkTimestamps.add(Instant.now()); // Record the timestamp of the detected blink
//                    if (blinkTimestamps.size() >= EYE_BLINK_THRESHOLD) {
//                        Instant latestBlink = blinkTimestamps.get(blinkTimestamps.size() - 1);
//                        Instant previousBlink = blinkTimestamps.get(blinkTimestamps.size() - EYE_BLINK_THRESHOLD);
//                        Duration timeDifference = Duration.between(previousBlink, latestBlink);
//                        if (timeDifference.toMillis() <= BLINK_TIME_THRESHOLD_MS) {
//                            isLive = true;
//                        }
//                    }
//                } else {
//                    blinkTimestamps.clear();
//                    isLive = false;
//                }
//
//                // Draw rectangles around the face and eyes
//                Imgproc.rectangle(frame, faceRect.tl(), faceRect.br(), new Scalar(0, 0, 255), 2);
//                for (Rect eyeRect : eyes.toArray()) {
//                    Rect absoluteEyeRect = new Rect(faceRect.x + eyeRect.x, faceRect.y + eyeRect.y,
//                            eyeRect.width, eyeRect.height);
//                    Imgproc.rectangle(frame, absoluteEyeRect.tl(), absoluteEyeRect.br(), new Scalar(0, 255, 0), 2);
//                }
//            }
//
//            // Display the frame with detected faces and eyes in the window
//            HighGui.imshow("Live Face Detection", frame);
//
//            // Check liveness status and display notification
//            if (isLive) {
//                System.out.println("Liveness: Real");
//            } else {
//                System.out.println("Liveness: Spoof");
//            }
//
//            // Exit the loop if the 'Esc' key is pressed
//            if (HighGui.waitKey(1) == 27)
//                break;
//        }
//
//        // Release the VideoCapture and close the window
//        videoCapture.release();
//        HighGui.destroyAllWindows();
//    }
//
//    private static void detectFaces(Mat frame, CascadeClassifier faceCascade, MatOfRect faces) {
//        Mat grayFrame = new Mat();
//        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.equalizeHist(grayFrame, grayFrame);
//        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//    }
//
//    private static void detectEyes(Mat faceROI, CascadeClassifier eyeCascade, MatOfRect eyes) {
//        Mat grayFaceROI = new Mat();
//        Imgproc.cvtColor(faceROI, grayFaceROI, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.equalizeHist(grayFaceROI, grayFaceROI);
//        eyeCascade.detectMultiScale(grayFaceROI, eyes, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//    }
//}

//import org.opencv.core.Core;
//import org.opencv.core.Mat;
//import org.opencv.core.MatOfRect;
//import org.opencv.core.Rect;
//import org.opencv.core.Scalar;
//import org.opencv.core.Size;
//import org.opencv.imgproc.Imgproc;
//import org.opencv.objdetect.CascadeClassifier;
//import org.opencv.objdetect.Objdetect;
//import org.opencv.videoio.VideoCapture;
//import org.opencv.videoio.Videoio;
//import org.opencv.highgui.HighGui;
//
//import java.time.Duration;
//import java.time.Instant;
//import java.util.ArrayList;
//import java.util.List;
//
//public class LiveFaceDetector {
//    private static final int EYE_BLINK_THRESHOLD = 3; // Number of consecutive eye blinks required for liveness detection
//    private static final long BLINK_TIME_THRESHOLD_MS = 300; // Minimum time difference between eye blinks (in milliseconds)
//
//    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
//        // Load the trained haarcascade classifier XML files for face and eye detection
//        CascadeClassifier faceCascade = new CascadeClassifier();
//        faceCascade.load("data/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_alt2.xml");
//
//        CascadeClassifier eyeCascade = new CascadeClassifier();
//        //eyeCascade.load("data/raw.githubusercontent.com_opencv_opencv_contrib_master_data_haarcascades_haarcascade_eye.xml");
//        eyeCascade.load("data/raw.githubusercontent.com_anaustinbeing_haar-cascade-files_master_haarcascade_eye.xml");
//
//        // Create a VideoCapture object to capture frames from the camera
//        VideoCapture videoCapture = new VideoCapture(0); // 0 represents the default camera
//
//        if (!videoCapture.isOpened()) {
//            System.out.println("Failed to open the camera.");
//            return;
//        }
//
//        // Set camera frame properties
//        videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 6400);
//        videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
//
//        // Create a window to display the camera feed
//        HighGui.namedWindow("Live Face Detection");
//
//        // Variables for eye blinking detection
//        List<Instant> blinkTimestamps = new ArrayList<>();
//        boolean isLive = false;
//
//        // Continuously read frames from the camera feed
//        Mat frame = new Mat();
//        while (true) {
//            // Read a frame from the camera
//            videoCapture.read(frame);
//
//            // Perform face detection on the frame
//            MatOfRect faces = new MatOfRect();
//            detectFaces(frame, faceCascade, faces);
//
//            // For each detected face, perform eye detection and check for blinking
//            Rect[] facesArray = faces.toArray();
//            for (Rect faceRect : facesArray) {
//                // Extract the region of interest (ROI) containing the face
//                Mat faceROI = frame.submat(faceRect);
//
//                // Detect eyes in the face ROI
//                MatOfRect eyes = new MatOfRect();
//                detectEyes(faceROI, eyeCascade, eyes);
//
//                // Check if eyes are detected and determine if the person is live based on eye blinking
//                if (eyes.toArray().length >= 1) {
//                    blinkTimestamps.add(Instant.now()); // Record the timestamp of the detected blink
//                    if (blinkTimestamps.size() >= EYE_BLINK_THRESHOLD) {
//                        Instant latestBlink = blinkTimestamps.get(blinkTimestamps.size() - 1);
//                        Instant previousBlink = blinkTimestamps.get(blinkTimestamps.size() - EYE_BLINK_THRESHOLD);
//                        Duration timeDifference = Duration.between(previousBlink, latestBlink);
//                        if (timeDifference.toMillis() <= BLINK_TIME_THRESHOLD_MS) {
//                            isLive = true;
//                        }
//                    }
//                } else {
//                    blinkTimestamps.clear();
//                    isLive = false;
//                }
//
//                // Draw rectangles around the face and eyes
//                Imgproc.rectangle(frame, faceRect.tl(), faceRect.br(), new Scalar(0, 0, 255), 2);
//                for (Rect eyeRect : eyes.toArray()) {
//                    Rect absoluteEyeRect = new Rect(faceRect.x + eyeRect.x, faceRect.y + eyeRect.y,
//                            eyeRect.width, eyeRect.height);
//                    Imgproc.rectangle(frame, absoluteEyeRect.tl(), absoluteEyeRect.br(), new Scalar(0, 255, 0), 2);
//                }
//            }
//
//            // Display the frame with detected faces and eyes in the window
//            HighGui.imshow("Live Face Detection", frame);
//
//            // Check liveness status and display notification
//            if (isLive) {
//                System.out.println("Liveness: Real");
//            } else {
//                System.out.println("Liveness: Spoof");
//            }
//
//            // Exit the loop if the 'Esc' key is pressed
//            if (HighGui.waitKey(1) == 27)
//                break;
//        }
//
//        // Release the VideoCapture and close the window
//        videoCapture.release();
//        HighGui.destroyAllWindows();
//    }
//
//    private static void detectFaces(Mat frame, CascadeClassifier faceCascade, MatOfRect faces) {
//        Mat grayFrame = new Mat();
//        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.equalizeHist(grayFrame, grayFrame);
//        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//    }
//
//    private static void detectEyes(Mat faceROI, CascadeClassifier eyeCascade, MatOfRect eyes) {
//        Mat grayFaceROI = new Mat();
//        Imgproc.cvtColor(faceROI, grayFaceROI, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.equalizeHist(grayFaceROI, grayFaceROI);
//        eyeCascade.detectMultiScale(grayFaceROI, eyes, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//    }
//}

//package com.opencv;
//
//import org.opencv.core.Core;
//import org.opencv.core.Mat;
//import org.opencv.core.MatOfRect;
//import org.opencv.core.Rect;
//import org.opencv.core.Scalar;
//import org.opencv.core.Size;
//import org.opencv.imgproc.Imgproc;
//import org.opencv.objdetect.CascadeClassifier;
//import org.opencv.objdetect.Objdetect;
//import org.opencv.videoio.VideoCapture;
//import org.opencv.videoio.Videoio;
//import org.opencv.highgui.HighGui;
//
//public class LiveFaceDetector {
//    private static final int EYE_BLINK_THRESHOLD = 3; // Number of consecutive eye blinks required for liveness detection
//
//    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
//        // Load the trained haarcascade classifier XML files for face and eye detection
//        CascadeClassifier faceCascade = new CascadeClassifier();
//        faceCascade.load("data/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_alt2.xml");
//
//        CascadeClassifier eyeCascade = new CascadeClassifier();
//        //eyeCascade.load("data/raw.githubusercontent.com_opencv_opencv_contrib_master_data_haarcascades_haarcascade_eye.xml");
//        eyeCascade.load("data/raw.githubusercontent.com_anaustinbeing_haar-cascade-files_master_haarcascade_eye.xml");
//
//        // Create a VideoCapture object to capture frames from the camera
//        VideoCapture videoCapture = new VideoCapture(0); // 0 represents the default camera
//
//        if (!videoCapture.isOpened()) {
//            System.out.println("Failed to open the camera.");
//            return;
//        }
//
//        // Set camera frame properties
//        videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
//        videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
//
//        // Create a window to display the camera feed
//        HighGui.namedWindow("Live Face Detection");
//
//        // Variables for eye blinking detection
//        int eyeBlinkCounter = 0;
//        boolean isLive = false;
//
//        // Continuously read frames from the camera feed
//        Mat frame = new Mat();
//        while (true) {
//            // Read a frame from the camera
//            videoCapture.read(frame);
//
//            // Perform face detection on the frame
//            MatOfRect faces = new MatOfRect();
//            detectFaces(frame, faceCascade, faces);
//
//            // For each detected face, perform eye detection and check for blinking
//            Rect[] facesArray = faces.toArray();
//            for (Rect faceRect : facesArray) {
//                // Extract the region of interest (ROI) containing the face
//                Mat faceROI = frame.submat(faceRect);
//
//                // Detect eyes in the face ROI
//                MatOfRect eyes = new MatOfRect();
//                detectEyes(faceROI, eyeCascade, eyes);
//
//                // Check if eyes are detected and determine if the person is live based on eye blinking
//                if (eyes.toArray().length >= 2) {
//                    eyeBlinkCounter++;
//                    if (eyeBlinkCounter >= EYE_BLINK_THRESHOLD) {
//                        isLive = true;
//                    }
//                } else {
//                    eyeBlinkCounter = 0;
//                    isLive = false;
//                }
//
//                // Draw rectangles around the face and eyes
//                Imgproc.rectangle(frame, faceRect.tl(), faceRect.br(), new Scalar(0, 0, 255), 2);
//                for (Rect eyeRect : eyes.toArray()) {
//                    Rect absoluteEyeRect = new Rect(faceRect.x + eyeRect.x, faceRect.y + eyeRect.y,
//                            eyeRect.width, eyeRect.height);
//                    Imgproc.rectangle(frame, absoluteEyeRect.tl(), absoluteEyeRect.br(), new Scalar(0, 255, 0), 2);
//                }
//            }
//
//            // Display the frame with detected faces and eyes in the window
//            HighGui.imshow("Live Face Detection", frame);
//
//            // Check liveness status and display notification
//            if (isLive) {
//                System.out.println("Liveness: Real");
//            } else {
//                System.out.println("Liveness: Spoof");
//            }
//
//            // Exit the loop if the 'Esc' key is pressed
//            if (HighGui.waitKey(1) == 27)
//                break;
//        }
//
//        // Release the VideoCapture and close the window
//        videoCapture.release();
//        HighGui.destroyAllWindows();
//    }
//
//    private static void detectFaces(Mat frame, CascadeClassifier faceCascade, MatOfRect faces) {
//        Mat grayFrame = new Mat();
//        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.equalizeHist(grayFrame, grayFrame);
//        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//    }
//
//    private static void detectEyes(Mat faceROI, CascadeClassifier eyeCascade, MatOfRect eyes) {
//        Mat grayFaceROI = new Mat();
//        Imgproc.cvtColor(faceROI, grayFaceROI, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.equalizeHist(grayFaceROI, grayFaceROI);
//        eyeCascade.detectMultiScale(grayFaceROI, eyes, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//    }
//}

//package com.opencv;
//
//import org.opencv.core.Core;
//import org.opencv.core.Mat;
//import org.opencv.core.MatOfRect;
//import org.opencv.core.Rect;
//import org.opencv.core.Scalar;
//import org.opencv.core.Size;
//import org.opencv.imgproc.Imgproc;
//import org.opencv.objdetect.CascadeClassifier;
//import org.opencv.objdetect.Objdetect;
//import org.opencv.videoio.VideoCapture;
//import org.opencv.videoio.Videoio;
//import org.opencv.highgui.HighGui;
//
//public class LiveFaceDetector {
//    private static final int EYE_BLINK_THRESHOLD = 3; // Number of consecutive eye blinks required for liveness detection
//
//    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
//        // Load the trained haarcascade classifier XML files for face and eye detection
//        CascadeClassifier faceCascade = new CascadeClassifier();
//        faceCascade.load("data/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_alt2.xml");
//
//        CascadeClassifier eyeCascade = new CascadeClassifier();
//        //eyeCascade.load("data/raw.githubusercontent.com_opencv_opencv_contrib_master_data_haarcascades_haarcascade_eye.xml");
//        eyeCascade.load("data/raw.githubusercontent.com_anaustinbeing_haar-cascade-files_master_haarcascade_eye.xml");
//
//        // Create a VideoCapture object to capture frames from the camera
//        VideoCapture videoCapture = new VideoCapture(0); // 0 represents the default camera
//
//        if (!videoCapture.isOpened()) {
//            System.out.println("Failed to open the camera.");
//            return;
//        }
//
//        // Set camera frame properties
//        videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
//        videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
//
//        // Create a window to display the camera feed
//        HighGui.namedWindow("Live Face Detection");
//
//        // Variables for eye blinking detection
//        int eyeBlinkCounter = 0;
//        boolean isLive = false;
//
//        // Continuously read frames from the camera feed
//        Mat frame = new Mat();
//        while (true) {
//            // Read a frame from the camera
//            videoCapture.read(frame);
//
//            // Perform face detection on the frame
//            MatOfRect faces = new MatOfRect();
//            detectFaces(frame, faceCascade, faces);
//
//            // For each detected face, perform eye detection and check for blinking
//            Rect[] facesArray = faces.toArray();
//            for (Rect faceRect : facesArray) {
//                // Extract the region of interest (ROI) containing the face
//                Mat faceROI = frame.submat(faceRect);
//
//                // Detect eyes in the face ROI
//                MatOfRect eyes = new MatOfRect();
//                detectEyes(faceROI, eyeCascade, eyes);
//
//                // Check if eyes are detected and determine if the person is live based on eye blinking
//                if (eyes.toArray().length >= 2) {
//                    eyeBlinkCounter++;
//                    if (eyeBlinkCounter >= EYE_BLINK_THRESHOLD) {
//                        isLive = true;
//                    }
//                } else {
//                    eyeBlinkCounter = 0;
//                    isLive = false;
//                }
//
//                // Draw rectangles around the face and eyes
//                Imgproc.rectangle(frame, faceRect.tl(), faceRect.br(), new Scalar(0, 0, 255), 2);
//                for (Rect eyeRect : eyes.toArray()) {
//                    Rect absoluteEyeRect = new Rect(faceRect.x + eyeRect.x, faceRect.y + eyeRect.y,
//                            eyeRect.width, eyeRect.height);
//                    Imgproc.rectangle(frame, absoluteEyeRect.tl(), absoluteEyeRect.br(), new Scalar(0, 255, 0), 2);
//                }
//            }
//
//            // Display the frame with detected faces and eyes in the window
//            HighGui.imshow("Live Face Detection", frame);
//
//            // Exit the loop if the 'Esc' key is pressed
//            if (HighGui.waitKey(1) == 27)
//                break;
//        }
//
//        // Release the VideoCapture and close the window
//        videoCapture.release();
//        HighGui.destroyAllWindows();
//    }
//
//    private static void detectFaces(Mat frame, CascadeClassifier faceCascade, MatOfRect faces) {
//        Mat grayFrame = new Mat();
//        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.equalizeHist(grayFrame, grayFrame);
//        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//    }
//
//    private static void detectEyes(Mat faceROI, CascadeClassifier eyeCascade, MatOfRect eyes) {
//        Mat grayFaceROI = new Mat();
//        Imgproc.cvtColor(faceROI, grayFaceROI, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.equalizeHist(grayFaceROI, grayFaceROI);
//        eyeCascade.detectMultiScale(grayFaceROI, eyes, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//    }
//}

//package com.opencv;
//
//import org.opencv.core.Core;
//import org.opencv.core.Mat;
//import org.opencv.core.MatOfRect;
//import org.opencv.core.Rect;
//import org.opencv.core.Scalar;
//import org.opencv.core.Size;
//import org.opencv.imgproc.Imgproc;
//import org.opencv.objdetect.CascadeClassifier;
//import org.opencv.objdetect.Objdetect;
//import org.opencv.videoio.VideoCapture;
//import org.opencv.videoio.Videoio;
//import org.opencv.highgui.HighGui;
//
//public class LiveFaceDetector {
//    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
//        // Load the trained haarcascade classifier XML file
//        CascadeClassifier faceCascade = new CascadeClassifier();
//        faceCascade.load("data/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_alt2.xml");
//
//        // Create a VideoCapture object to capture frames from the camera
//        VideoCapture videoCapture = new VideoCapture(0); // 0 represents the default camera
//
//        if (!videoCapture.isOpened()) {
//            System.out.println("Failed to open the camera.");
//            return;
//        }
//
//        // Set camera frame properties
//        videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
//        videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
//
//        // Create a window to display the camera feed
//        HighGui.namedWindow("Live Face Detection");
//
//        // Continuously read frames from the camera feed
//        Mat frame = new Mat();
//        while (true) {
//            // Read a frame from the camera
//            videoCapture.read(frame);
//
//            // Perform face detection on the frame
//            detectAndDrawFaces(frame, faceCascade);
//
//            // Display the frame with detected faces in the window
//            HighGui.imshow("Live Face Detection", frame);
//
//            // Exit the loop if the 'Esc' key is pressed
//            if (HighGui.waitKey(1) == 27)
//                break;
//        }
//
//        // Release the VideoCapture and close the window
//        videoCapture.release();
//        HighGui.destroyAllWindows();
//    }
//
//    private static void detectAndDrawFaces(Mat frame, CascadeClassifier faceCascade) {
//        // Convert the frame to grayscale
//        Mat grayFrame = new Mat();
//        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
//
//        // Equalize the histogram of the grayscale frame
//        Imgproc.equalizeHist(grayFrame, grayFrame);
//
//        // Detect faces in the frame
//        MatOfRect faces = new MatOfRect();
//        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//                new Size(30, 30), new Size());
//
//        // Draw rectangles around the detected faces
//        Rect[] facesArray = faces.toArray();
//        for (Rect faceRect : facesArray) {
//            Imgproc.rectangle(frame, faceRect.tl(), faceRect.br(), new Scalar(0, 0, 255), 2);
//        }
//    }
//}

//package com.opencv;
//
//import org.opencv.core.Core;
//import org.opencv.core.Mat;
//import org.opencv.core.Rect;
//import org.opencv.core.Scalar;
//import org.opencv.core.Size;
//import org.opencv.imgcodecs.Imgcodecs;
//import org.opencv.imgproc.Imgproc;
//import org.opencv.objdetect.CascadeClassifier;
//import org.opencv.objdetect.Objdetect;
//import org.opencv.core.MatOfRect;
//
//public class FaceDetector {
//    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        Mat image = Imgcodecs.imread("C:\\Users\\mumar\\Downloads\\PRI_223554170.webp");
//
//        // create method for detect and save
//        detectAndSave(image);
//    }
//
//    private static void detectAndSave(Mat image) {
//
//        // create some objects
//        Mat grayFrame = new Mat();
//        Imgproc.cvtColor(image, grayFrame, Imgproc.COLOR_BGR2GRAY);
//
//        // improve contrast for better result
//        Imgproc.equalizeHist(grayFrame, grayFrame);
//
//        int height = grayFrame.height();
//        int absoluteFacesize = 0;
//        if (Math.round(height * 0.2f) > 0) {
//            absoluteFacesize = Math.round(height * 0.2f);
//        }
//
//        // Detect faces
//        CascadeClassifier faceCascade = new CascadeClassifier();
//
//        // load trained data file
//        faceCascade.load("data/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_alt2.xml");
//
//        MatOfRect faces = new MatOfRect();
//        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, 
//        		new Size(absoluteFacesize, absoluteFacesize), new Size());
//
//        // write to file
//        Rect[] faceArray = faces.toArray();
//        for (int i = 0; i < faceArray.length; i++) {
//            // draw rect
//            Imgproc.rectangle(image, faceArray[i], new Scalar(0, 0, 255), 3);
//        }
//
//        Imgcodecs.imwrite("images/output.jpg", image);
//        System.out.println("Write success: " + faceArray.length);
//    }
//}