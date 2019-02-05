"""
Placeholder flask app for http requests, use server.py instead
"""
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from model_work import Model
from PIL import Image
import io
import numpy as np
import base64
import skimage.io
import re
import scipy.misc
import matplotlib.pyplot as plt

app = Flask(__name__)
cors = CORS(app)
segnet_stream = 'weak'
ckpt_name = 'vgg_16_4chan_' + segnet_stream
ckpt_path = '/home/ubuntu/dev/segment/weak_simple_model/' + ckpt_name + '/' + ckpt_name + '.ckpt-' + str(50000)
model = Model(ckpt_path)
x1 = 42
y1 =  30
x2 = x1 + 557
y2 = y1+ 369
bbox = [y1, x1, y2, x2]

@app.route('/', methods=['POST'])
def handle_my_event():
    print("DATA :: {}".format(request.data))
    image_b64 = request.data
    #img_str = re.search(r'data:image/png;base64,(.*)',image_b64).group(1)
    #imgdata = base64.b64decode(img_str)
    input_image = skimage.io.imread(image_b64, plugin='imageio')
    input_image = input_image[:, :, :3]
    plt.imsave('test_img.png', input_image)
    input_image = scipy.misc.imresize(input_image, (399, 600))
    inp_mask = model.rect_mask((input_image.shape[0], input_image.shape[1], 1), bbox)
    inp_mask_new = np.concatenate((input_image, inp_mask), axis=-1)
    inp_mask_new = np.expand_dims(inp_mask_new, axis=0)
    print(inp_mask_new.shape)
    masks = model.test(inp_mask_new, ckpt_path, '.', '.', 'weak')
    print(masks)
    return ('my-response', {'masks': base64.b64encode(masks)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888)
