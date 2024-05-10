import cv2  # Import the OpenCV library
import mediapipe as mp  # Import the Mediapipe library
import datetime  # Import the datetime module
import tkinter as tk  # Import the tkinter library
from PIL import Image, ImageTk  # Import the Image and ImageTk modules from the PIL library
from tkinter import ttk  # Import the ttk module from the tkinter library
from Model.HandPoseAnalyze import HandPoseAnalyzer  # Import the HandPoseAnalyzer class from the HandPoseAnalyze module
from Model.HandPoseDetector import h_gesture  # Import the h_gesture function from the HandPoseDetector module
from Model.HandPoseDetector import alignment_detection  # Import the alignment_detection function from the HandPoseDetector module
import threading
import cv2
import time
from tkinter.font import Font


class VideoCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            time.sleep(0.01)  # 稍微延时减轻CPU压力

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


class HandGestureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("800x600")  # Set the initial window size to 800x600 to fit the layout
        self.minsize(800, 600)  # Set the minimum window size to 800x600
        self.title("Hand Gesture Detector")  # Set the window title to "Hand Gesture Detector"
        self.init_ui()  # Initialize the user interface
        self.running = False  # Initialize the running state to False
        self.init_camera()  # Initialize the camera
    def init_ui(self):
        # Create a label frame for the video display area
        self.video_frame = ttk.LabelFrame(self, text="Video Display")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)
        self.canvas_width, self.canvas_height = 640, 480
        self.video_canvas = tk.Canvas(self.video_frame, width=self.canvas_width, height=self.canvas_height)
        self.video_canvas.pack()

        # Create a frame for the control buttons
        self.control_frame = ttk.Frame(self)
        self.control_frame.grid(row=0, column=1, sticky=tk.N)

        # Create start and stop buttons
        self.start_button = ttk.Button(self.control_frame, text="START", command=self.start_detection)
        self.stop_button = ttk.Button(self.control_frame, text="STOP", command=self.stop_detection)

        self.start_button.pack(fill=tk.X, pady=5)
        self.stop_button.pack(fill=tk.X, pady=5)
        
        large_font = Font(size=16)
        # Create an output text area
        self.output_text = tk.Text(self, height=15, width=100)
        self.output_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.output_text.tag_configure('large_font', font=large_font)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

    def init_camera(self):

        self.camera = VideoCapture(0)
        # Initialize the hand detector in the Mediapipe library
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.update_frame()

    def stop_detection(self):
        self.running = False
        self.camera.release()  # Release the camera
        self.output_text.delete('1.0', tk.END)
       

    def print_info(self, text):
        self.output_text.insert(tk.END, text + "\n")

    def update_frame(self):
        if self.running:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                results = self.hands.process(frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        hand_local = [(landmark.x * frame.shape[1], landmark.y * frame.shape[0]) for landmark in
                                      hand_landmarks.landmark]
                        angle_list, distance_48, distance_37, distance_26, distance_812, distance_1216, distance_1620,left_hand = HandPoseAnalyzer.hand_angle(hand_local)
                        gesture_result = h_gesture(angle_list, distance_48, distance_37, distance_26, distance_812, distance_1216, distance_1620,left_hand)  # 假设distance_48是第一个
                        alignment_result = alignment_detection(angle_list, distance_48)
                        self.display_gesture_results(gesture_result, alignment_result)
                self.display_frame(frame)
                self.after(10, self.update_frame)  # 继续更新帧

    def display_gesture_results(self, gesture_result, alignment_result):
    # 清空现有文本
      self.output_text.delete('1.0', tk.END)

    # 插入文本并应用大号字体
      self.output_text.insert(tk.END, 'Gesture Analysis: ', 'large_font')
      self.output_text.insert(tk.END, f"{gesture_result}\n", 'large_font')
      self.output_text.insert(tk.END, 'Alignment Suggestions: ', 'large_font')
      self.output_text.insert(tk.END, f"{alignment_result}\n", 'large_font')
      
    def display_frame(self, frame):
        frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
        frame = Image.fromarray(frame)
        frame_image = ImageTk.PhotoImage(image=frame)
        if hasattr(self, 'canvas_image_id'):
            self.video_canvas.itemconfig(self.canvas_image_id, image=frame_image)
        else:
            self.canvas_image_id = self.video_canvas.create_image(0, 0, anchor='nw', image=frame_image)
        self.video_canvas.imgtk = frame_image  # Keep the reference


if __name__ == "__main__":
    app = HandGestureApp()
    app.mainloop()  # Run the main program loop

