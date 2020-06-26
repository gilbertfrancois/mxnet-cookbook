# Faster R-CNN Notes

<img src="/Users/gilbert/Development/git/mxnet-cookbook/_resources/bike-and-bus.jpg" style="zoom:50%;" />

*Input image*

## Feature extractor

**Resnet 50 v1** with **bottlenecks**. Output feature space is $(N, 1024, w//stride, h//stride)$, where $stride=16$.

![](/Users/gilbert/Development/git/mxnet-cookbook/_resources/faster_rcnn_feat.png)

```
self.features.summary(x)
--------------------------------------------------------------------------------
        Layer (type)                                Output Shape         Param #
================================================================================
               Input                            (1, 3, 600, 800)               0
            Conv2D-1                           (1, 64, 300, 400)            9408
         BatchNorm-2                           (1, 64, 300, 400)             256
        Activation-3                           (1, 64, 300, 400)               0
         MaxPool2D-4                           (1, 64, 150, 200)               0
            Conv2D-5                           (1, 64, 150, 200)            4096
         BatchNorm-6                           (1, 64, 150, 200)             256
        Activation-7                           (1, 64, 150, 200)               0
            Conv2D-8                           (1, 64, 150, 200)           36864
         BatchNorm-9                           (1, 64, 150, 200)             256
       Activation-10                           (1, 64, 150, 200)               0
           Conv2D-11                          (1, 256, 150, 200)           16384
        BatchNorm-12                          (1, 256, 150, 200)            1024
           Conv2D-13                          (1, 256, 150, 200)           16384
        BatchNorm-14                          (1, 256, 150, 200)            1024
       Activation-15                          (1, 256, 150, 200)               0
    BottleneckV1b-16                          (1, 256, 150, 200)               0
           Conv2D-17                           (1, 64, 150, 200)           16384
        BatchNorm-18                           (1, 64, 150, 200)             256
       Activation-19                           (1, 64, 150, 200)               0
           Conv2D-20                           (1, 64, 150, 200)           36864
        BatchNorm-21                           (1, 64, 150, 200)             256
       Activation-22                           (1, 64, 150, 200)               0
           Conv2D-23                          (1, 256, 150, 200)           16384
        BatchNorm-24                          (1, 256, 150, 200)            1024
       Activation-25                          (1, 256, 150, 200)               0
    BottleneckV1b-26                          (1, 256, 150, 200)               0
           Conv2D-27                           (1, 64, 150, 200)           16384
        BatchNorm-28                           (1, 64, 150, 200)             256
       Activation-29                           (1, 64, 150, 200)               0
           Conv2D-30                           (1, 64, 150, 200)           36864
        BatchNorm-31                           (1, 64, 150, 200)             256
       Activation-32                           (1, 64, 150, 200)               0
           Conv2D-33                          (1, 256, 150, 200)           16384
        BatchNorm-34                          (1, 256, 150, 200)            1024
       Activation-35                          (1, 256, 150, 200)               0
    BottleneckV1b-36                          (1, 256, 150, 200)               0
           Conv2D-37                          (1, 128, 150, 200)           32768
        BatchNorm-38                          (1, 128, 150, 200)             512
       Activation-39                          (1, 128, 150, 200)               0
           Conv2D-40                           (1, 128, 75, 100)          147456
        BatchNorm-41                           (1, 128, 75, 100)             512
       Activation-42                           (1, 128, 75, 100)               0
           Conv2D-43                           (1, 512, 75, 100)           65536
        BatchNorm-44                           (1, 512, 75, 100)            2048
           Conv2D-45                           (1, 512, 75, 100)          131072
        BatchNorm-46                           (1, 512, 75, 100)            2048
       Activation-47                           (1, 512, 75, 100)               0
    BottleneckV1b-48                           (1, 512, 75, 100)               0
           Conv2D-49                           (1, 128, 75, 100)           65536
        BatchNorm-50                           (1, 128, 75, 100)             512
       Activation-51                           (1, 128, 75, 100)               0
           Conv2D-52                           (1, 128, 75, 100)          147456
        BatchNorm-53                           (1, 128, 75, 100)             512
       Activation-54                           (1, 128, 75, 100)               0
           Conv2D-55                           (1, 512, 75, 100)           65536
        BatchNorm-56                           (1, 512, 75, 100)            2048
       Activation-57                           (1, 512, 75, 100)               0
    BottleneckV1b-58                           (1, 512, 75, 100)               0
           Conv2D-59                           (1, 128, 75, 100)           65536
        BatchNorm-60                           (1, 128, 75, 100)             512
       Activation-61                           (1, 128, 75, 100)               0
           Conv2D-62                           (1, 128, 75, 100)          147456
        BatchNorm-63                           (1, 128, 75, 100)             512
       Activation-64                           (1, 128, 75, 100)               0
           Conv2D-65                           (1, 512, 75, 100)           65536
        BatchNorm-66                           (1, 512, 75, 100)            2048
       Activation-67                           (1, 512, 75, 100)               0
    BottleneckV1b-68                           (1, 512, 75, 100)               0
           Conv2D-69                           (1, 128, 75, 100)           65536
        BatchNorm-70                           (1, 128, 75, 100)             512
       Activation-71                           (1, 128, 75, 100)               0
           Conv2D-72                           (1, 128, 75, 100)          147456
        BatchNorm-73                           (1, 128, 75, 100)             512
       Activation-74                           (1, 128, 75, 100)               0
           Conv2D-75                           (1, 512, 75, 100)           65536
        BatchNorm-76                           (1, 512, 75, 100)            2048
       Activation-77                           (1, 512, 75, 100)               0
    BottleneckV1b-78                           (1, 512, 75, 100)               0
           Conv2D-79                           (1, 256, 75, 100)          131072
        BatchNorm-80                           (1, 256, 75, 100)            1024
       Activation-81                           (1, 256, 75, 100)               0
           Conv2D-82                            (1, 256, 38, 50)          589824
        BatchNorm-83                            (1, 256, 38, 50)            1024
       Activation-84                            (1, 256, 38, 50)               0
           Conv2D-85                           (1, 1024, 38, 50)          262144
        BatchNorm-86                           (1, 1024, 38, 50)            4096
           Conv2D-87                           (1, 1024, 38, 50)          524288
        BatchNorm-88                           (1, 1024, 38, 50)            4096
       Activation-89                           (1, 1024, 38, 50)               0
    BottleneckV1b-90                           (1, 1024, 38, 50)               0
           Conv2D-91                            (1, 256, 38, 50)          262144
        BatchNorm-92                            (1, 256, 38, 50)            1024
       Activation-93                            (1, 256, 38, 50)               0
           Conv2D-94                            (1, 256, 38, 50)          589824
        BatchNorm-95                            (1, 256, 38, 50)            1024
       Activation-96                            (1, 256, 38, 50)               0
           Conv2D-97                           (1, 1024, 38, 50)          262144
        BatchNorm-98                           (1, 1024, 38, 50)            4096
       Activation-99                           (1, 1024, 38, 50)               0
   BottleneckV1b-100                           (1, 1024, 38, 50)               0
          Conv2D-101                            (1, 256, 38, 50)          262144
       BatchNorm-102                            (1, 256, 38, 50)            1024
      Activation-103                            (1, 256, 38, 50)               0
          Conv2D-104                            (1, 256, 38, 50)          589824
       BatchNorm-105                            (1, 256, 38, 50)            1024
      Activation-106                            (1, 256, 38, 50)               0
          Conv2D-107                           (1, 1024, 38, 50)          262144
       BatchNorm-108                           (1, 1024, 38, 50)            4096
      Activation-109                           (1, 1024, 38, 50)               0
   BottleneckV1b-110                           (1, 1024, 38, 50)               0
          Conv2D-111                            (1, 256, 38, 50)          262144
       BatchNorm-112                            (1, 256, 38, 50)            1024
      Activation-113                            (1, 256, 38, 50)               0
          Conv2D-114                            (1, 256, 38, 50)          589824
       BatchNorm-115                            (1, 256, 38, 50)            1024
      Activation-116                            (1, 256, 38, 50)               0
          Conv2D-117                           (1, 1024, 38, 50)          262144
       BatchNorm-118                           (1, 1024, 38, 50)            4096
      Activation-119                           (1, 1024, 38, 50)               0
   BottleneckV1b-120                           (1, 1024, 38, 50)               0
          Conv2D-121                            (1, 256, 38, 50)          262144
       BatchNorm-122                            (1, 256, 38, 50)            1024
      Activation-123                            (1, 256, 38, 50)               0
          Conv2D-124                            (1, 256, 38, 50)          589824
       BatchNorm-125                            (1, 256, 38, 50)            1024
      Activation-126                            (1, 256, 38, 50)               0
          Conv2D-127                           (1, 1024, 38, 50)          262144
       BatchNorm-128                           (1, 1024, 38, 50)            4096
      Activation-129                           (1, 1024, 38, 50)               0
   BottleneckV1b-130                           (1, 1024, 38, 50)               0
          Conv2D-131                            (1, 256, 38, 50)          262144
       BatchNorm-132                            (1, 256, 38, 50)            1024
      Activation-133                            (1, 256, 38, 50)               0
          Conv2D-134                            (1, 256, 38, 50)          589824
       BatchNorm-135                            (1, 256, 38, 50)            1024
      Activation-136                            (1, 256, 38, 50)               0
          Conv2D-137                           (1, 1024, 38, 50)          262144
       BatchNorm-138                           (1, 1024, 38, 50)            4096
      Activation-139                           (1, 1024, 38, 50)               0
   BottleneckV1b-140                           (1, 1024, 38, 50)               0
================================================================================
Parameters in forward computation graph, duplicate included
   Total params: 8573888
   Trainable params: 8543296
   Non-trainable params: 30592
Shared params in forward computation graph: 0
Unique parameters in model: 8573888
--------------------------------------------------------------------------------

```





## RPN

```
self.rpn.summary(F.zeros_like(x), feat[0])
--------------------------------------------------------------------------------
        Layer (type)                                Output Shape         Param #
================================================================================
               Input         (1, 3, 600, 800), (1, 1024, 38, 50)               0
RPNAnchorGenerator-1                               (1, 28500, 4)          983040
            Conv2D-2                           (1, 1024, 38, 50)         9438208
        Activation-3                           (1, 1024, 38, 50)               0
            Conv2D-4                             (1, 15, 38, 50)           15375
            Conv2D-5                             (1, 60, 38, 50)           61500
NormalizedBoxCenterDecoder-6                       (1, 28500, 4)               0
   BBoxClipToImage-7                               (1, 28500, 4)               0
       RPNProposal-8                               (1, 28500, 5)               0
               RPN-9                    (1, 300, 1), (1, 300, 4)               0
================================================================================
Parameters in forward computation graph, duplicate included
   Total params: 10498123
   Trainable params: 9515083
   Non-trainable params: 983040
Shared params in forward computation graph: 0
Unique parameters in model: 10498123
--------------------------------------------------------------------------------

```

```
RPN(
  (anchor_generator): RPNAnchorGenerator(
  
  )
  (conv1): HybridSequential(
    (0): Conv2D(1024 -> 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Activation(relu)
  )
  (score): Conv2D(1024 -> 15, kernel_size=(1, 1), stride=(1, 1))
  (loc): Conv2D(1024 -> 60, kernel_size=(1, 1), stride=(1, 1))
  (region_proposer): RPNProposal(
    (_box_to_center): BBoxCornerToCenter(
    
    )
    (_box_decoder): NormalizedBoxCenterDecoder(
      (corner_to_center): BBoxCornerToCenter(
      
      )
    )
    (_clipper): BBoxClipToImage(
    
    )
  )
)
```

```
self.top_features
Out[83]: 
HybridSequential(
  (0): HybridSequential(
    (0): BottleneckV1b(
      (conv1): Conv2D(1024 -> 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=512)
      (relu1): Activation(relu)
      (conv2): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=512)
      (relu2): Activation(relu)
      (conv3): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=2048)
      (relu3): Activation(relu)
      (downsample): HybridSequential(
        (0): Conv2D(1024 -> 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=2048)
      )
    )
    (1): BottleneckV1b(
      (conv1): Conv2D(2048 -> 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=512)
      (relu1): Activation(relu)
      (conv2): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=512)
      (relu2): Activation(relu)
      (conv3): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=2048)
      (relu3): Activation(relu)
    )
    (2): BottleneckV1b(
      (conv1): Conv2D(2048 -> 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=512)
      (relu1): Activation(relu)
      (conv2): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=512)
      (relu2): Activation(relu)
      (conv3): Conv2D(512 -> 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=True, in_channels=2048)
      (relu3): Activation(relu)
    )
  )
)
```

![](/Users/gilbert/Development/git/mxnet-cookbook/_resources/faster_rcnn_rpn_output.png)

Output of the region proposal network. Only boxes with $score \geq 0.5$ are plotted for clarity.

![](/Users/gilbert/Development/git/mxnet-cookbook/_resources/faster_rcnn_rpn_bboxes_output.png)

The RPN boxes that will be selected eventually.

```
pooled_feat.shape
Out[81]: (300, 1024, 14, 14)
self.top_features.summary(pooled_feat)
--------------------------------------------------------------------------------
        Layer (type)                                Output Shape         Param #
================================================================================
               Input                         (300, 1024, 14, 14)               0
            Conv2D-1                          (300, 512, 14, 14)          524288
         BatchNorm-2                          (300, 512, 14, 14)            2048
        Activation-3                          (300, 512, 14, 14)               0
            Conv2D-4                            (300, 512, 7, 7)         2359296
         BatchNorm-5                            (300, 512, 7, 7)            2048
        Activation-6                            (300, 512, 7, 7)               0
            Conv2D-7                           (300, 2048, 7, 7)         1048576
         BatchNorm-8                           (300, 2048, 7, 7)            8192
            Conv2D-9                           (300, 2048, 7, 7)         2097152
        BatchNorm-10                           (300, 2048, 7, 7)            8192
       Activation-11                           (300, 2048, 7, 7)               0
    BottleneckV1b-12                           (300, 2048, 7, 7)               0
           Conv2D-13                            (300, 512, 7, 7)         1048576
        BatchNorm-14                            (300, 512, 7, 7)            2048
       Activation-15                            (300, 512, 7, 7)               0
           Conv2D-16                            (300, 512, 7, 7)         2359296
        BatchNorm-17                            (300, 512, 7, 7)            2048
       Activation-18                            (300, 512, 7, 7)               0
           Conv2D-19                           (300, 2048, 7, 7)         1048576
        BatchNorm-20                           (300, 2048, 7, 7)            8192
       Activation-21                           (300, 2048, 7, 7)               0
    BottleneckV1b-22                           (300, 2048, 7, 7)               0
           Conv2D-23                            (300, 512, 7, 7)         1048576
        BatchNorm-24                            (300, 512, 7, 7)            2048
       Activation-25                            (300, 512, 7, 7)               0
           Conv2D-26                            (300, 512, 7, 7)         2359296
        BatchNorm-27                            (300, 512, 7, 7)            2048
       Activation-28                            (300, 512, 7, 7)               0
           Conv2D-29                           (300, 2048, 7, 7)         1048576
        BatchNorm-30                           (300, 2048, 7, 7)            8192
       Activation-31                           (300, 2048, 7, 7)               0
    BottleneckV1b-32                           (300, 2048, 7, 7)               0
================================================================================
Parameters in forward computation graph, duplicate included
   Total params: 14987264
   Trainable params: 14964736
   Non-trainable params: 22528
Shared params in forward computation graph: 0
Unique parameters in model: 14987264
--------------------------------------------------------------------------------

```

## Class and box predictors

```
self.class_predictor.summary(box_feat)
--------------------------------------------------------------------------------
        Layer (type)                                Output Shape         Param #
================================================================================
               Input                           (300, 2048, 1, 1)               0
             Dense-1                                   (300, 21)           43029
================================================================================
Parameters in forward computation graph, duplicate included
   Total params: 43029
   Trainable params: 43029
   Non-trainable params: 0
Shared params in forward computation graph: 0
Unique parameters in model: 43029
--------------------------------------------------------------------------------

```

```
self.class_predictor
Out[84]: Dense(2048 -> 21, linear)
```



```
self.box_predictor
Out[85]: Dense(2048 -> 80, linear)
```

```
self.box_predictor.summary(box_feat)
--------------------------------------------------------------------------------
        Layer (type)                                Output Shape         Param #
================================================================================
               Input                           (300, 2048, 1, 1)               0
             Dense-1                                   (300, 80)          163920
================================================================================
Parameters in forward computation graph, duplicate included
   Total params: 163920
   Trainable params: 163920
   Non-trainable params: 0
Shared params in forward computation graph: 0
Unique parameters in model: 163920
--------------------------------------------------------------------------------
```

![](/Users/gilbert/Development/git/mxnet-cookbook/_resources/faster_rcnn_bboxes.png)

Final prediction.