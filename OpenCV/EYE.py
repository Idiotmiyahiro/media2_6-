#EYE:Python側

import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import serial


# 歩行者用信号機かチェックし、歩行者用信号機であればその画像を返す
def ReturnPedestrianTrafficLight(results):

    # 結果が0でないことを確認する(ただし、resultsに格納されているのはtraffic lightのみ)
    #if len(results.crop()) == 0:
        #return None

    # 何かしらの信号機が認識されている場合に以下のコードが実行される
    max_area = 0
    max_img = None

    for traffic_light in results.crop():    # 全ての画像に適用
        img = traffic_light['im']
        img_shape = img.shape

        # 縦長画像であれば良い
        if img_shape[0] > img_shape[1]:
            # トリミング面積を計算
            area = img_shape[0] * img_shape[1]
            if area > max_area:
                max_area = area
                max_img = img

            return max_img
        else:
            continue

# 上下から色判定領域を抽出する関数
def extractRedBlueArea(img):
    img_shape = img.shape
    w_c = int(img_shape[1] / 2)
    s = int(img_shape[1] / 6)
    # 上（赤色領域）のエリアにおける抽出画像の中心点を設定
    upper_h_c = int(img_shape[0] / 4)
    # 下（青色領域）のエリアにおける抽出画像の中心点を設定
    lower_h_c = int(img_shape[0] * 3 / 4)

    return [img[upper_h_c - s:upper_h_c + s, w_c - s:w_c+s, :], img[lower_h_c - s:lower_h_c + s, w_c - s:w_c+s, :]]

# 上側の画像と下側の画像をリストにしたものを引数として渡す
def ReturnTrafficLightSignal(img_list):
    
    # 画像データはBGRとする
    upper_img = img_list[0]
    lower_img = img_list[1]

    # 上側（赤）のランプの状態を検出
    upper_red_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:,:,0].mean()   # 赤色成分の平均値
    upper_blue_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:,:,2].mean()  # 青色成分の平均値
    # 差分を求める
    upper_delta = abs(upper_red_nums - upper_blue_nums)

    # 下側（青）のランプの状態を検出
    lower_red_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:,:,0].mean()   # 赤色成分の平均値
    lower_blue_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:,:,2].mean()  # 青色成分の平均値
    # 差分を求める
    lower_delta = abs(lower_red_nums - lower_blue_nums)

    if upper_delta >= lower_delta:
        return 'r'
    else:
        return "b" 
    


# 学習済みモデルの読み込み
model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.save_results = False  # 結果を保存しないようにする

# GPUで実行する場合
#device = 'cuda:0'
# クラスインデックスを設定
model.classes = [0]



def main():
    #カメラを起動（通常は0またはカメラのデバイス番号を指定
    cap = cv2.VideoCapture(1)
    
    with serial.Serial('COM5', 9600, timeout=1) as ser:
        while True:
            # カメラからフレームを読み込む
            ret, frame = cap.read()
            # フレームが正しく読み取れたかを確認
            if not ret:
                print("Failed to read frame from camera.")
                break
            
            # iVcamから取得したRGB画像をBGR形式に変換
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

            # YOLOv5モデルにフレームを渡して検出を行う
            results = model(frame_bgr)

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
                    print("Pedestrian Traffic Light Color: ", color)

                    # Arduinoに信号を送信
                    flag = bytes(color, 'utf-8')
                    ser.write(flag)

            # フレームを表示
            cv2.imshow('Pedestrian Traffic Light Detection', frame)

            # 'q'キーを押したらループを抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # カメラとウィンドウを解放
    cap.release()
    cv2.destroyAllWindows()
            
            
    if __name__ == "__main__":
    # YOLOv5モデルの読み込み
    model_path = 'best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    # GPUで実行する場合
    device = 'cuda:0'
    # クラスインデックスを設定
    model.classes = [0]
    
    model.save_results = False  # 結果を保存しないようにする

    # main関数の呼び出し
    main()


