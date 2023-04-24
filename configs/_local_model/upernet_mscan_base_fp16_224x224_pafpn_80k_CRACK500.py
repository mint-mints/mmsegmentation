_base_ = [
    '../_base_/models/upernet_mscan.py', '../_base_/datasets/CRACK500.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (224, 224)
model = dict(
    decode_head=dict(type='UPerPAFPNHead',
                     loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
                                  dict(type='DiceLoss', loss_name='loss_dice', loss_weight=4.0)],
                     dropout_ratio=0.7
                     ),
    auxiliary_head=dict(loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
                                     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=4.0)],
                        dropout_ratio=0.7
                        ),
    test_cfg=dict(mode='whole'),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

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

evaluation = dict(interval=8000, metric=['mIoU', 'mDice', 'mFscore'], save_best='mIoU', pre_eval=True)


# +------------+-------+-------+-------+--------+-----------+--------+
# |   Class    |  IoU  |  Acc  |  Dice | Fscore | Precision | Recall |
# +------------+-------+-------+-------+--------+-----------+--------+
# | background |  97.5 |  98.9 | 98.74 | 98.74  |   98.57   |  98.9  |
# |   crack    | 64.27 | 76.07 | 78.25 | 78.25  |   80.55   | 76.07  |
# +------------+-------+-------+-------+--------+-----------+--------+
# 2023-04-20 06:15:15,935 - mmseg - INFO - Summary:
# 2023-04-20 06:15:15,935 - mmseg - INFO -
# +-------+-------+-------+-------+---------+------------+---------+
# |  aAcc |  mIoU |  mAcc | mDice | mFscore | mPrecision | mRecall |
# +-------+-------+-------+-------+---------+------------+---------+
# | 97.61 | 80.88 | 87.48 | 88.49 |  88.49  |   89.56    |  87.48  |
# +-------+-------+-------+-------+---------+------------+---------+