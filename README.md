# FaceDetector

## Task

To detect face and classify emotion

## Dataset 

The number of training images is 1,688.

### RGB Mean & Standard Deviation
Mean: [0.5747293, 0.53332627, 0.5069782]
Standard Deviation: [0.23505978, 0.23163822, 0.233239]

### Category(#num)
- Neutral(573)
- Anger(379)
- Surprise(226)
- Smile(627)
- Sad(311)

### Image Resolution
- Width Range: [113, 7296]
- Height Range: [114, 4870]
- Median Width: 400
- Median Height: 326

I set the input image size of the network to 384x384 via median width and height.

## Network

### Backbone Network
I adopted RexNet as backbone network to design a lightweight detector.

### Main Architecture
Feature Pyramid Network + YOLOv3(anchor based detection method)

I found that the state of the art methods for the face detection exploited the architecture of the feature pyramid network.

[Pyramid Box A Context-assisted Single Shot Face Detector](https://arxiv.org/pdf/1803.07737v2.pdf)

So I referenced the architecture of feature pyramid network to design a scale robust network.

#### Anchor Boxes
Used 3 anchor boxes

### Model Size
42.8 MB

## Training
- Epochs: 200
- Batch Size: 32
- Optimizer: SGD with Momentum
- Augmentation: Random Translation, Random Cropping, Random Scaling(x0.5 to x2.0)

## Commands

### Training
```
python train.py
```

### Test
[trained weights download](https://drive.google.com/file/d/1lfaSXKLqf0lxgLnGXRJIrLBNM7RDF8an/view?usp=sharing)
```
python test.py --weights best_nota_face_detector_200.pth
```

### Results

<img src="./figures/1.JPG" width="80%">
<img src="./figures/2.jpg" width="80%">
<img src="./figures/3.JPG" width="80%">
<img src="./figures/4.JPG" width="80%">
<img src="./figures/5.jpg" width="80%">
<img src="./figures/6.jpg" width="80%">
<img src="./figures/7.JPG" width="80%">
<img src="./figures/8.jpg" width="80%">
<img src="./figures/9.jpg" width="80%">
<img src="./figures/10.JPG" width="80%">
<img src="./figures/11.jpg" width="80%">
<img src="./figures/12.jpg" width="80%">





