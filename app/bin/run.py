from __future__ import unicode_literals
import os, hashlib
from flask import Flask, request, abort, send_from_directory, render_template
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, StickerMessage, ImageMessage, ImageSendMessage, RichMenu
import configparser

from shutil import copyfile
from PIL import Image
import numpy as np

import tensorflow as tf
print("tensorflow version:", tf.__version__)
from tensorflow.python.keras.backend import set_session
import keras
from keras.models import load_model
from keras import activations
from keras.preprocessing import image
from vis.visualization import overlay,visualize_cam
from vis.utils import utils

server_path = 'https://dac8-111-235-248-163.ngrok.io/'
# server_path = 'https://api.openaifab.com:15003/'
linebot_web_hook = 'https://developers.line.biz/console/channel/1613908363/messaging-api'
hash_salt = 'hello_2330'

# Set path
folder_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # <app> folder
template_dir = os.path.join(folder_path, "html", "templates")
static_dir = os.path.join(folder_path, "html", "static")
media_dir = os.path.join(folder_path, "media")
model_path = os.path.join(folder_path, "model", "pd_normal_vgg16_88_bg.h5")

app = Flask(__name__, template_folder=template_dir, static_url_path='/static', static_folder=static_dir)

config = configparser.ConfigParser()
config.read(os.path.join(folder_path, "bin", 'config.ini'))
line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        print(body, signature)
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請傳一張手繪螺旋圖"))


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # create dir
    user_id = hashlib.sha224((event.source.user_id+hash_salt).encode()).hexdigest()
    timestamp = str(event.timestamp)
    save_dir = os.path.join(media_dir, user_id)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    output_dir = os.path.join(static_dir, user_id)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # save image
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)
    save_name = os.path.join(save_dir,timestamp+'.jpg')
    output_reply_path = "static/"+user_id+"/"+timestamp+'.jpg'
    with open(save_name, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)
    print('----------recieve image done----------')
    # inference
    result = p_detection(save_name, os.path.join(static_dir,user_id,timestamp+'.jpg'))
    # copyfile(os.path.join(save_name), os.path.join(static_dir, user_id, timestamp+'.jpg'))
    # result = "測試中"
    print('----------AI inference done-----------')
    # reply LINE message
    line_bot_api.reply_message(
        event.reply_token,
        [
            TextSendMessage(
                text = "小帕AI預測："+result
            ),
            ImageSendMessage(
                original_content_url = server_path+output_reply_path,
                preview_image_url = server_path+output_reply_path
            )
        ]
    )


#parkinson detection
def p_detection(input_path, output_path):
    # load input image and preprocess
    img_RGB = image.load_img(input_path, target_size=(256,256))
    img_RGB = image.img_to_array(img_RGB)
    # print("resized image shape:", img_RGB.shape)
    x = np.expand_dims(img_RGB, axis=0) / 255.0
    # print("shape of the x for visualize_cam:", x.shape)
    # classification model inference
    with gGraph.as_default():
        set_session(gSess)
        # classifer 2 classes
        classes = gModel.predict(x)
        print("predicted result:", classes)
        if classes[0] <=0.5:
            class_name = labels[0]
        else:
            class_name = labels[1]
        print('predicted class name is:',class_name)
        # Generate heatmap of the predicted image
        heatmap = visualize_cam(gModel_h, -1, filter_indices=None, seed_input=x[0,:,:,:])
        # print("shape of heatmap:",heatmap.shape)
    # combine two pic
    img = Image.fromarray(overlay(img_RGB, heatmap).astype('uint8'))
    img.save(output_path)
    return class_name


@app.route('/', methods=['GET'])
def run_app():
    return render_template('index.html')


if __name__ == "__main__":
    # Set labels
    labels = {0:'正常',1:'有帕金森的風險'}
    ### use session and graph for flask
    gSess = tf.Session()
    gGraph = tf.get_default_graph()
    set_session(gSess)
    # load classification model
    gModel = load_model(model_path)
    # CAM model
    # Modify the last layer of the model to make CAM model
    gModel_h = gModel
    print("Remove Activation from Last Layer")
    gModel_h.layers[-1].activation = activations.linear
    print("Now Applying changes to the model ...")
    gModel_h = utils.apply_modifications(gModel_h)
    print("==============================")
    print("classification model loaded!!!")
    print("==============================")

    # app.run(debug=True, host='0.0.0.0', port=15003, use_reloader=False, ssl_context='adhoc')
    # app.run(debug=True, host='0.0.0.0', port=15003, use_reloader=False, ssl_context=(os.path.join(folder_path, "html/ssl/openaifab.com/fullchain4.pem"), os.path.join(folder_path, "html/ssl/openaifab.com/privkey4.pem")))
    app.run(debug=True, host='0.0.0.0', port=15003, use_reloader=False) # use_reloader=False
