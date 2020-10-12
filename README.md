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

I referenced the feature pyramid network to design a scale robust network.

#### Anchors
Used 3 anchor boxes

## Training
- Epochs: 200
- Batch Size: 32
- Optimizer: SGD with Momentum
- Augmentation: Random Translation, Random Cropping, Random Scaling(x0.5 to x2.0)

