import cv2
import time

# カメラを指定します。通常、0番目のカメラがデフォルトです。
camera_index = 2

# VideoCaptureのインスタンスを作成します。
cap = cv2.VideoCapture(camera_index)

# カメラが正常に開かれたかを確認します。
if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

# フレームレートの初期化
fps = 0
frame_count = 0
start_time = time.time()

while True:
    # カメラから1フレーム読み込みます。
    ret, frame = cap.read()

    # フレームの読み込みに成功した場合
    if ret:
        # フレームを表示します。
        cv2.imshow("Web Camera", frame)


        # 1秒ごとにフレームレートを計算してファイルに追記
        frame_count += 1
        if time.time() - start_time >= 1.0:
            fps = frame_count / (time.time() - start_time)
            with open("frame_rate.txt", 'a') as f:
                print(f"フレームレート: {fps:.2f} FPS", file=f)
            start_time = time.time()
            frame_count = 0


    # 'q'キーを押すとループを終了します。
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラキャプチャを解放し、ウィンドウを閉じます。
cap.release()
cv2.destroyAllWindows()