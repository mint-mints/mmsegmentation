_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/crack_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100epoch.py'
]
crop_size = (224, 224)
model = dict(
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=2,
                     loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
                                  dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
                     dropout_ratio=0.7
                     ),
    auxiliary_head=dict(in_channels=512, num_classes=2,
                        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
                                     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
                        dropout_ratio=0.7
                        ),
    test_cfg=dict(mode='whole'),
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()

load_from = 'checkpoints/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth'
