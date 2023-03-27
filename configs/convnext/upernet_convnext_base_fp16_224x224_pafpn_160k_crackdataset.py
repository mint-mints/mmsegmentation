_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/crack_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (224, 224)
model = dict(
    decode_head=dict(type='UPerPAFPNHead',
                     in_channels=[128, 256, 512, 1024], num_classes=2,
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

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()

load_from = 'checkpoints/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth'
