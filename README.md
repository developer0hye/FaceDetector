# FaceDetector

## Task

To detect face and classify emotion

## Dataset 

The number of training images is 1,688.

### Category(#num)

- neutral(573)
- anger(379)
- surprise(226)
- smile(627)
- sad(311)

### Image Resolution
- Width Range: [113, 7296]
- Height Range: [114, 4870]
- Median Width: 400
- Median Height: 326

I set the input image size of the network to 416x416 via median width and height.

## Network

### Backbone Network
I adopted RexNet to design a lightweight detector as backbone network.

