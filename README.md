# UMMKD
**Unpaired Multi-modal Segmentation with Knowledge Distillation**

> We propose a novel learning scheme for unpaired cross-modality image segmentation, with a highly compact architecture achieving superior segmentation accuracy. In our method, we heavily reuse network parameters, by sharing all convolutional kernels across CT and MRI, and only employ modality-specific internal normalization layers which compute respective statistics. To effectively train such a highly compact model, we introduce a novel loss term inspired by knowledge distillation, by explicitly constraining the KL-divergence of our derived prediction distributions between modalities. We have extensively validated our approach on two multi-class segmentation problems: i) cardiac structure segmentation, and ii) abdominal organ segmentation. Different network settings, i.e., 2D dilated network and 3D U-Net, are utilized to investigate our method's general efficacy. 


## Running UMMKD
To run our 2D version, can directly use the tfrecord data released in our another relevant project from [here](https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation)

To train the model, specify the training configurations (can simply use the default setting), and run:
```
python main_combine.py  --phase training
```

To test the model, specify the path of the model to be tested, and run:
```
python main_combine.py  --phase testing
```
