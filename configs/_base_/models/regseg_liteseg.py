# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='RegSeg',
        name='exp48_decoder26',
        num_classes=2,
        ),
    decode_head=dict(
        type='PPLiteSegHead',
        in_channels=[48, 128, 320],
        in_index=[0, 1, 2],
        # feature_strides=[4, 8, 16],
        channels=32,
        num_classes=2,
        norm_cfg=norm_cfg,
        # align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
