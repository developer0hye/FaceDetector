# FaceDetector

## Task

To detect face and classify emotion

## Dataset 

The number of training images is 1,688.

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

### Objects Size
- Width Range: [18, 3668]
- Height Range: [25, 3668]
- Median Width: 83
- Median Height: 108

#### Visualizaiton




## Network

### Backbone Network
I adopted RexNet to design a lightweight detector as backbone network.

