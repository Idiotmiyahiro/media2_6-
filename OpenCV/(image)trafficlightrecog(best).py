import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

# 歩行者用信号機かチェックし、歩行者用信号機であればその画像を返す
def ReturnPedestrianTrafficLight(results):

    # 結果が0でないことを確認する(ただし、resultsに格納されているのはtraffic lightのみ)
    if len(results.crop()) == 0:
        return None

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
        # 歩行者用信号機だったら、その部分を切り抜いた画像を返す
        
        else:
            return None   # 歩行者用信号機ではなかったらNoneを返す
        
        
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
"""def ReturnTrafficLightSignal(img_list):
    
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
        return "b" """


def ReturnTrafficLightSignal(img_list):
    # 画像データはBGRとする
    upper_img = img_list[0]
    lower_img = img_list[1]

    # 上側（赤）のランプの状態を検出
    upper_sum = np.sum(upper_img, axis=(0, 1))
    upper_sum_all = np.sum(upper_sum)  # BGR成分の総和

    # 下側（青）のランプの状態を検出
    lower_sum = np.sum(lower_img, axis=(0, 1))
    lower_sum_all = np.sum(lower_sum)  # BGR成分の総和

    if upper_sum_all >= lower_sum_all:
        return 'r'
    else:
        return 'b'


#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#model.classes = [9]
#model.save_results = False  # 結果を保存しないようにする

# 学習済みモデルの読み込み
model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)


# CPUで実行する場合
#device = 'cpu'
# GPUで実行する場合
#device = 'cuda:0'

# クラスインデックスを設定
model.classes = [0]


def TrafficLightSignal(imgArray_or_imgPath):
    results = model(imgArray_or_imgPath)
    img = ReturnPedestrianTrafficLight(results)

    if type(img) == np.ndarray:  # 歩行者用信号機の画像があるか確かめる
        RedBlueImgs = extractRedBlueArea(img)
    else:
        return None

    return ReturnTrafficLightSignal(RedBlueImgs)

save_img=False
img_path = 'data/images/uu.png'
result=TrafficLightSignal(img_path)
print(result)