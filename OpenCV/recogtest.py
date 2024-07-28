import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import yaml


# ReturnPedestrianTrafficLight 関数内でクラスのインデックスを取得
def ReturnPedestrianTrafficLight(results, class_names=['pedestrian-traffic-light', 'pedestrian-traffic-light-non-']):
    # 対象クラスが検出されているか確認する
    for target_class in class_names:
        if target_class in class_names:
            class_index = class_names.index(target_class)
            
            # 修正：クラスのインデックスを使用する
            for det in results.xyxy[0]:  # 全ての検出結果に適用
                if det[-1] == class_index:  # 対象クラスの場合
                    img = results.imgs[0][det[0].int().item()]  # 対象クラスの画像を取得
                    img_shape = img.shape
                    
            # 結果が0でないことを確認する
            if len(results.xyxy[0]) == 0:
                continue

            # 何かしらの信号機が認識されている場合に以下のコードが実行される
            max_area = 0
            max_img = None

            for det in results.xyxy[0]:  # 全ての検出結果に適用
                if det[-1] == class_index:  # 対象クラスの場合
                    img = results.imgs[0][det[0].int().item()]  # 対象クラスの画像を取得
                    img_shape = img.shape

                    # 縦長画像であれば良い
                    if img_shape[0] > img_shape[1]:
                        # トリミング面積を計算
                        area = img_shape[0] * img_shape[1]
                        if area > max_area:
                            max_area = area
                            max_img = img

            return max_img, target_class

    return None, None


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
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)


# model.yaml ファイルからクラス情報を読み込む
with open('data.yaml', 'r') as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

# クラス情報を取得
class_names = model_config['names']


# CPUで実行する場合
# device = 'cpu'
# GPUで実行する場合
device = 'cuda:0'

# クラスインデックスを設定
model.classes = [0, 1]  # 0: pedestrian_traffic_light, 1: pedestrian_traffic_light(non)


def TrafficLightSignal(imgArray_or_imgPath):
    results = model(imgArray_or_imgPath)
    
    # 'pedestrian_traffic_light' が検出された場合のみ画像処理を行う
    img, processed_class = ReturnPedestrianTrafficLight(results, model.names)
    
    if img is not None:  # 歩行者用信号機の画像があるか確かめる
        RedBlueImgs = extractRedBlueArea(img)
    else:
        return None

    return ReturnTrafficLightSignal(RedBlueImgs), processed_class

save_img = False
img_path = 'traffic154.jpeg'
result, processed_class = TrafficLightSignal(img_path)
print(result)
print(f"Processed class: {processed_class}")
