# Google Driveをマウント（接続・利用可能状態にする
from google.colab import drive

drive.mount('/content/gdrive')

# imshow（ディスプレイへ画像を出力するコマンド）サポートパッチのインポート
from google.colab.patches import cv2_imshow

# dnn用（OpenCV・ディープラーニングの推論機能を担当するモジュール）
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

# インストール
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import IPython
from google.colab import output
from PIL import Image
from io import BytesIO
import base64
import urllib.request

print("ローディング")
prototxt = '/content/gdrive/My Drive/deploy.prototxt'
model = '/content/gdrive/My Drive/res10_300x300_ssd_iter_140000.caffemodel'

prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt";
model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel";

urllib.request.urlretrieve(prototxt_url, prototxt)
urllib.request.urlretrieve(model_url, model)

net = cv2.dnn.readNetFromCaffe(prototxt, model)

def face_detection(_img):
	# 幅640画素になるようにリサイズする
	_img = imutils.resize(_img, width=640)
	(h, w) = _img.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# 物体検出器にblobを適用する
	# print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# confidenceが0.5を超える領域をマーク
	# print(detections.shape[2])
	for i in range(0, detections.shape[2]):

  	# ネットワークが出力したconfidenceの値を抽出する
		confidence = detections[0, 0, i, 2]

  	# confidenceの値が0.5以上の領域のみを検出結果として描画する
		if confidence > 0.5:
			# 対象領域のバウンディングボックスの座標を計算する
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# バウンディングボックスとconfidenceの値を描画する
			text = "accuracy: {:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(_img, (startX, startY), (endX, endY), (255, 0, 255), 2)
			cv2.putText(_img, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

	return _img

def run(img_str):
    #decode to image
    decimg = base64.b64decode(img_str.split(',')[1], validate=True) #ヒント：前述の英語
    decimg = Image.open(BytesIO(decimg))
    decimg = np.array(decimg, dtype=np.uint8);
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

    out_img = face_detection(decimg)

    #encode to string
    _, encimg = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    encimg = encimg.tobytes()
    img_str = "data:image/jpeg;base64," + base64.b64encode(encimg).decode('utf-8') #ヒント：前述の英語
    return IPython.display.JSON({'img_str': img_str})

output.register_callback('notebook.run', run)

def use_cam(quality=0.8):
  js = Javascript('''
    async function useCam(quality) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      //映像要素
      const video = document.createElement('video');
      video.style.display = 'None';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      //ディスプレイ表示設定（フレームレートはディスプレイサイズやJPEGクオリティに依存）
      display_size = 500
      const src_canvas = document.createElement('canvas');
      src_canvas.width  = display_size;
      src_canvas.height = display_size * video.videoHeight / video.videoWidth;
      const src_canvasCtx = src_canvas.getContext('2d');
      src_canvasCtx.translate(src_canvas.width, 0);
      src_canvasCtx.scale(-1, 1);
      div.appendChild(src_canvas);

      const dst_canvas = document.createElement('canvas');
      dst_canvas.width  = src_canvas.width;
      dst_canvas.height = src_canvas.height;
      const dst_canvasCtx = dst_canvas.getContext('2d');
      div.appendChild(dst_canvas);

      //ビデオ停止ボタン
      const btn_div = document.createElement('div');
      document.body.appendChild(btn_div);
      const exit_btn = document.createElement('button');
      exit_btn.textContent = 'VideoStop';
      var exit_flg = true
      exit_btn.onclick = function() {exit_flg = false};
      btn_div.appendChild(exit_btn);

      // アウトプットをビデオ要素にフィット・リサイズさせる
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      var send_num = 0
      // ループ設定
      _canvasUpdate();
      async function _canvasUpdate() {
            src_canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, src_canvas.width, src_canvas.height);
            if (send_num<1){
                send_num += 1
                const img = src_canvas.toDataURL('image/jpeg', quality);
                const result = google.colab.kernel.invokeFunction('notebook.run', [img], {});
                result.then(function(value) {
                    parse = JSON.parse(JSON.stringify(value))["data"]
                    parse = JSON.parse(JSON.stringify(parse))["application/json"]
                    parse = JSON.parse(JSON.stringify(parse))["img_str"]
                    var image = new Image()
                    image.src = parse;
                    image.onload = function(){dst_canvasCtx.drawImage(image, 0, 0)}
                    send_num -= 1
                })
            }
            if (exit_flg){
                requestAnimationFrame(_canvasUpdate);
            }else{
                stream.getVideoTracks()[0].stop();
            }
      };
    }
    ''')
  display(js)
  data = eval_js('useCam({})'.format(quality))

use_cam()
