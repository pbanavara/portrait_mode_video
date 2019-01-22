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

let modelPromise;
let baseModel = 'lite_mobilenet_v2';

window.onload = () => modelPromise = cocoSsd.load();
var streaming = false;
var width = 600;    // We will scale the photo width to this
var height = 0;
  
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
// CHange later to video using image for the time being
//const video = document.getElementById('video');
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
    //context.drawImage(video, 0, 0, canvas.width, canvas.height);
    let baseModel = 'lite_mobilenet_v2';
    const model = await modelPromise;
    const result = await model.detect(image);
    console.log(result[0].bbox);
    for (let i=0;i<result.length;++i) {
	var data = context.getImageData(0, 0, image.width, image.height).data;
	blurPixels(data, result[i].bbox, image.width);
	context.beginPath();
	context.rect(...result[i].bbox);
	context.lineWidth = 1;
	context.strokeStyle = 'green';
	context.fillStyle='green';
	context.stroke();
    }
  }
      
function takePicture(img) {
    var context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    var data = canvas.toDataURL('image/jpeg');
    var img = new Image();
    img.src = data;
    processImage(img).then(results => console.log(results));
    setTimeout(function () {
	takePicture();
    }, 0);
}

function blurPixels(data, bbox, width) {
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
