import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np



def ReturnPedestrianTrafficLight(results):
    for traffic_light in results.crop():
        img = traffic_light['im']
        img_shape = img.shape
        
        if img_shape[0] > img_shape[1]:
            return img
        else:
            continue

def extractRedBlueArea(img):
    img_shape = img.shape
    w_c = int(img_shape[1] / 2)
    s = int(img_shape[1] / 6)
    upper_h_c = int(img_shape[0] / 4)
    lower_h_c = int(img_shape[0] * 3 / 4)
    return [img[upper_h_c - s:upper_h_c + s, w_c - s:w_c+s, :], img[lower_h_c - s:lower_h_c + s, w_c - s:w_c+s, :]]

def ReturnTrafficLightSignal(img_list):
    upper_img = img_list[0]
    lower_img = img_list[1]
    upper_red_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:,:,0].mean()
    upper_blue_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:,:,2].mean()
    upper_delta = abs(upper_red_nums - upper_blue_nums)
    lower_red_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:,:,0].mean()
    lower_blue_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:,:,2].mean()
    lower_delta = abs(lower_red_nums - lower_blue_nums)
    if upper_delta >= lower_delta:
        return 'r'
    else:
        return 'b'

# YOLOv5モデルの読み込み
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s')
model.classes = [9]  # クラスインデックスを設定
model.save_results = False  # 結果を保存しないようにする

# カメラを起動（通常は0またはカメラのデバイス番号を指定します）
cap = cv2.VideoCapture(0)

while True:
    # カメラからフレームを読み込む
    ret, frame = cap.read()

    # YOLOv5モデルにフレームを渡して検出を行う
    results = model(frame)

    # 歩行者用信号機の検出結果を取得
    pedestrian_traffic_light = results.pred[0]

    # 検出された信号機があれば処理を行う
    if len(pedestrian_traffic_light) > 0:
        # 信号機の色を判定
        img = ReturnPedestrianTrafficLight(results)
        if img is not None:
            RedBlueImgs = extractRedBlueArea(img)
            color = ReturnTrafficLightSignal(RedBlueImgs)
            # 信号機の色を表示
            print("Pedestrian Traffic Light Color: ", color)  # 'r'または'b'が表示される

    # フレームを表示
    cv2.imshow('Pedestrian Traffic Light Detection', frame)

    # 'q'キーを押したらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラとウィンドウを解放
cap.release()
cv2.destroyAllWindows()
