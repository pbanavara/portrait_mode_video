/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as cocoSsd from '@tensorflow-models/coco-ssd'

//Global variables
let modelPromise;
import io from 'socket.io-client';
import * as http from 'http';
import * as uploader from 'base64-image-upload';

window.onload = () => modelPromise = cocoSsd.load();
var streaming = false;
var width = 600;    // We will scale the photo width to this
var height = 0;
//Temporary variable to store previous frame values
var tempBbox;
var tempBytes;
var outputImage = new Image();
function hasGetUserMedia() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}
if (hasGetUserMedia()) {
  // Good to go!
} else {
  alert('getUserMedia() is not supported by your browser');
}
const constraints = {
    video: true
};
const video = document.querySelector('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext("2d");
navigator.mediaDevices.getUserMedia(constraints).
    then((stream) => {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log("Error in streaming");
    });
video.addEventListener('canplay', function(ev){
    if (!streaming) {
        height = video.videoHeight / (video.videoWidth/width);
        video.setAttribute('width', width);
        video.setAttribute('height', height);
        canvas.setAttribute('width', width);
        canvas.setAttribute('height', height);
            streaming = true;
        takePicture();
    }
}, false);

async function processImage(image) {
    /**
     * Async function to process images emanating from a video stream
     * @param image a Javascript image object
     */
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const model = await modelPromise;
    const result = await model.detect(image);
    for (let i=0;i<result.length;++i) {
        var data = context.getImageData(0, 0, image.width, image.height).data;
        changeBackgroundPixels(data, result[i].bbox, image.width);
        context.beginPath();
        context.rect(...result[i].bbox);
        context.lineWidth = 1;
        context.strokeStyle = 'green';
        context.fillStyle='green';
        context.stroke();
    }
    return result;
}

function encode (input) {
    /**
     * A helpler method to encode the bytearray image for rendering in the context
     */
    var keyStr = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
    var output = "";
    var chr1, chr2, chr3, enc1, enc2, enc3, enc4;
    var i = 0;
    while (i < input.length) {
        chr1 = input[i++];
        chr2 = i < input.length ? input[i++] : Number.NaN; // Not sure if the index 
        chr3 = i < input.length ? input[i++] : Number.NaN; // checks are needed here

        enc1 = chr1 >> 2;
        enc2 = ((chr1 & 3) << 4) | (chr2 >> 4);
        enc3 = ((chr2 & 15) << 2) | (chr3 >> 6);
        enc4 = chr3 & 63;

        if (isNaN(chr2)) {
            enc3 = enc4 = 64;
        } else if (isNaN(chr3)) {
            enc4 = 64;
        }
        output += keyStr.charAt(enc1) + keyStr.charAt(enc2) +
                  keyStr.charAt(enc3) + keyStr.charAt(enc4);
    }
    return output;
}

function takePicture(img) {
    /**
     * Method to obtain semantic sergmentation masks from the backend server
     * Calls the process image helper method asynchronously to get the processed mask
     * @param img A javascript image object to be processed 
     */
    var context = canvas.getContext('2d');
    var data = canvas.toDataURL('image/jpeg');
    var data_sock = canvas.toDataURL();
    var img = new Image();
    img.src = data;
    processImage(img).then(results => {
        console.log(results);
        if (results.length <= 0) {
            throw "The bounding box was not returned";
        }
        var bbox = results[0].bbox;
        if (tempBytes == undefined) {
            tempBbox = results[0].bbox; 
            const socket = io('http://54.88.230.253:8888');
            socket.on('connect', function() {
            socket.emit('my event', {'data_new': data_sock, 'bbox': tempBbox},
                    function(error, resp) {
                    console.log(error);
                    console.log(resp);
                    });
            });
            socket.on('my-response', function(data){
            console.log(data.masks);   //should output 'hello world'
            var arrayBuffer = data.masks;
            var bytes = new Uint8Array(arrayBuffer);
            tempBytes = bytes;
            var encodedBytes = encode(bytes);
            outputImage.onload = function() {
                context.drawImage(outputImage, 0, 0, canvas.width, canvas.height);
            }
            outputImage.src = 'data:image/png;base64,'+ encodedBytes;
            });
        } else if (Math.abs(tempBbox[0] - bbox[0]) < 15) {
            // Do not call the segment mask API, just render the previous image bytes
            var encodedBytes = encode(tempBytes);
            outputImage.onload = function() {
                context.drawImage(outputImage, 0, 0, canvas.width, canvas.height);
            }
            outputImage.src = 'data:image/png;base64,'+ encodedBytes;
        }
    });
    setTimeout(function () {
        /**
         * This is a crude hack there has to be a better way to process the video frames
         */
        takePicture();
    }, 2000);
}

function changeBackgroundPixels(data, bbox, width) {
    /*
      Convert all pixels outside the bounding box to yellow
      @param: data - Image data
      @param: bbox - the Bounding box returned from the cocossd model
      @param width - canvas width
    */
    console.log(data.length);
    console.log(bbox);
    for (var i = 0; i < data.length; i += 4) {
    var x = (i / 4) % width;
        var y = Math.floor((i / 4) / width);
        if (x < bbox[0] || x > bbox[0]+bbox[2] || y < bbox[1] || y > bbox[1]+bbox[3]) {
            //console.log("Blur image");
            context.fillStyle = "rgba("+255+","+255+","+0+","+(255)+")";
            context.fillRect(x, y, 1, 1);
        }
    }
}
