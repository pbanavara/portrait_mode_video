"""
Websocket based backend to load and serve the semantic segmentation model
"""

from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from model_work import Model
from PIL import Image
import io
import numpy as np
import base64
import skimage.io
import re
import scipy.misc

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
segnet_stream = 'weak'
ckpt_name = 'vgg_16_4chan_' + segnet_stream
ckpt_path = '/home/ubuntu/dev/segment/weak_simple_model/' + ckpt_name + '/' + ckpt_name + '.ckpt-' + str(50000)
model = Model(ckpt_path)

@socketio.on('my event')
def handle_my_event(arg1):
    """
    Socket io implementation for incoming video streams
    @param: arg1 - a json object containing keys 'data' the byte encoded image
    and 'bbox' the bounding box for that image
    @return: Socket emit on a response channel, the image with the background masked
    """
    bb = arg1['bbox']
    x1 = int(bb[0])
    y1 =  int(bb[1])
    x2 = x1 + int(bb[2])
    y2 = y1+ int(bb[3])
    bbox = [y1, x1, y2, x2]
    image_b64 = arg1['data_new']
    img_str = re.search(r'data:image/png;base64,(.*)',image_b64).group(1)
    imgdata = base64.b64decode(img_str)
    input_image = skimage.io.imread(imgdata, plugin='imageio')
    input_image = input_image[:, :, :3]
    
    input_image = scipy.misc.imresize(input_image, (399, 600))
    inp_mask = model.rect_mask((input_image.shape[0], input_image.shape[1], 1), bbox)
    inp_mask_new = np.concatenate((input_image, inp_mask), axis=-1)
    inp_mask_new = np.expand_dims(inp_mask_new, axis=0)
    masks = model.test(inp_mask_new, ckpt_path, '.', '.', 'weak')
    final_image = draw_mask(input_image, masks[0][:, :, 0], [0, 0, 0])
    ret_image = Image.fromarray(final_image).convert("RGB")
    imgByteArr = io.BytesIO()
    ret_image.save(imgByteArr, format='PNG')
    ret_image.save('out2.png', format='PNG')
    imgByteArr = imgByteArr.getvalue()
    emit ('my-response', {'masks': imgByteArr})

def draw_mask(image, mask, color, alpha=0.9, in_place=False):
    """
    Mask the background with black pixels ( a placeholder for gaussian blur)
    @param: image - numpy array of the RGB image
    @param: mask - numpy array of mask pixels
    @return: masked image
    """
    mask = np.expand_dims(mask, axis=-1)
    threshold = (np.max(mask) - np.min(mask)) / 2
    multiplier = 1 if np.amax(color) > 1 else 255
    masked_image = image if in_place == True else np.copy(image)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask[:,:,0] < threshold,
                                         masked_image[:, :, c] *
                                         (1 - alpha) + alpha * color[c] * multiplier,
                                         masked_image[:, :, c])

    return masked_image

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8888)
