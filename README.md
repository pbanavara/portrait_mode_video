## Bloorio
### Backgroiund blur for video conferencing

### I have used a hacky 2 stage approach to blur the background in videos

* Use object detection to obtain a bounding box
* Use the bounding box as a prior to 'cut' the image into foreground and background using Grabcut variant algorithms

### Installation instructions

Requires Python 3.5+ ( Has not been tested on Python 3.7) and a Tensorflow GPU
Please use a Conda package manager whenever possible.

#### Model
Download the pre-trained checkpoint files from [This drive location](http://bit.ly/tf-wss)
```
pip install -r requirements.txt
```

#### Backend
```
cd src/stage4_semantic_seg/
gunicorn --worker-class eventlet -w 1 server:app -b 0.0.0.0:8888
```
#### Frontend(Browser)
```
cd src/stage4_semantic_seg/client
yarn
yarn watch
```

### Results

* Stage 1 object detection
[Image](./images/obj_detect.png)
* Stage 2 Semantic segmentation
[Image](./images/semantic_seg.png)

### References
* Paper [Simple Does It: Weakly Supervised Instance and Semantic Segmentation](https://arxiv.org/abs/1603.07485)
* [Weakly supervised semantic segmentation with Tensorflow](https://github.com/philferriere/tfwss)
