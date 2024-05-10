import numpy as np
import pandas as pd
import joblib

loaded_model = joblib.load('random_forest_model.pkl')  # 这里是模型的保存地址

def h_gesture(angle_list, distance_48, distance_37, distance_26, distance_812, distance_1216, distance_1620,left_hand):
    data = {
    'Thumb Angle': [angle_list[0]],
    'Index Angle': [angle_list[1]],
    'Middle Angle': [angle_list[2]],
    'Ring Angle': [angle_list[3]],
    'Pinky Angle': [angle_list[4]],
    'Angle 5': [angle_list[5]],
    'Angle 6': [angle_list[6]],
    'Angle 7': [angle_list[7]],
    'Angle 8': [angle_list[8]],
    'Angle 9': [angle_list[9]],
    'Angle 10': [angle_list[10]],
    'Angle 11': [angle_list[11]],
    'Distance 48': [distance_48],
    'Distance 37': [distance_37],
    'Distance 26': [distance_26],
    'distance_812': [distance_812],
    'distance_1216': [distance_1216],
    'distance_1620': [distance_1620],
    'Hand_Left': [left_hand],
    'Hand_Right': [not left_hand]
    }
    new_data = pd.DataFrame(data)

    gesture_str = "Wrong Posture"
    if 65535. not in angle_list:
        if loaded_model.predict(new_data) == 1:
            gesture_str = "Correct Posture"

    return gesture_str

# Used to detect incorrect hand postures
def alignment_detection(angle_list, distance_48):
    min_correct_tumb = 30
    max_correct_tumb = 100
    min_correct_index = 80
    max_correct_index = 180
    min_correct_middle = 90
    max_correct_middle = 180
    min_correct_ring = 80
    max_correct_ring = 180
    min_correct_pinky = 80
    max_correct_pinky = 180
    max_dis48 = 100
    detection_result = ""

    if angle_list[0] < min_correct_tumb:
        detection_result += "Thumb angle too small.（拇指角度太小） "
    elif angle_list[0] > max_correct_tumb:
        detection_result += "Thumb angle too large.（拇指角度太大） "

    if angle_list[1] < min_correct_index:
        detection_result += "Index finger angle too small. （食指角度太小）"
    if angle_list[2] < min_correct_middle:
        detection_result += "Middle finger angle too small.（中指角度太小） "
    if angle_list[3] < min_correct_ring:
        detection_result += "Ring finger angle too small.（无名指角度太小） "
    if angle_list[4] < min_correct_pinky:
        detection_result += "Pinky finger angle too small. （小指角度太小）"

    if distance_48 > max_dis48:
        detection_result += "Thumb and index finger distance too large.（大拇指和食指距离太长） "

    return detection_result
