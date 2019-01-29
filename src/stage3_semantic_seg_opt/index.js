import * as tf from '@tensorflow/tfjs';

/*
import imageURL from './image3.jpg';
const image = document.getElementById('image');
image.src = imageURL;

const runButton = document.getElementById('run');
console.log("Loaded");
runButton.onclick = async () => {
*/
    /*
    const model = await tf.loadFrozenModel(
	'http://localhost:8080/tensorflowjs_model.pb',
	'http://localhost:8080/weights_manifest.json');
    const result = await model.predict(image);
    console.log(result);
    */
const runButton = document.getElementById('run');
console.log("Loaded");
runButton.onclick = async () => {
    const MODEL_URL = 'http://localhost:8080/tensorflowjs_model.pb';
    const WEIGHTS_URL = 'http://localhost:8080/weights_manifest.json';
    // Model's input and output have width and height of 513.
    const TENSOR_EDGE = 513;
    const [model, stream] = await Promise.all([
        tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL),
        navigator.mediaDevices.getUserMedia({video: {facingMode: 'user',
						     frameRate: 30, width : 640,
						     height:480}})]);
    const video = document.createElement('video');
    video.autoplay = true;
    video.width = video.height = TENSOR_EDGE;
    const ctx = document.getElementById("canvas").getContext("2d");       
    const videoCopy = ctx.canvas.cloneNode(false).getContext("2d");
    const maskContext = document.createElement('canvas').getContext("2d");
    maskContext.canvas.width = maskContext.canvas.height = TENSOR_EDGE;
    const img = maskContext.createImageData(TENSOR_EDGE, TENSOR_EDGE);
    let imgd = img.data;
      new Uint32Array(imgd.buffer).fill(0x00FFFF00);

    const render = () => {
      videoCopy.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);
      const out = tf.tidy(() => {
        return model.execute({'ImageTensor': tf.fromPixels(video).expandDims(0)});
      });
      const data = out.dataSync();
      for (let i = 0; i < data.length; i++) {
        imgd[i * 4 + 3] = data[i] == 15 ? 0 : 255;
      }
      maskContext.putImageData(img, 0, 0);
      ctx.drawImage(videoCopy.canvas, 0, 0);
      if (document.getElementById("show-background-toggle").checked)
        ctx.drawImage(maskContext.canvas, 0, 0, ctx.canvas.width,
		      ctx.canvas.height);
      window.requestAnimationFrame(render);
    }
    video.oncanplay = render;
    video.srcObject = stream;
 };


