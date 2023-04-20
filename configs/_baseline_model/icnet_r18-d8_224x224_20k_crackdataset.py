_base_ = [
    '../_base_/models/icnet_r50-d8.py',
    '../_base_/datasets/crack_dataset.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    backbone=dict(layer_channels=(128, 512), backbone_cfg=dict(depth=18)),
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))

crop_size = (224, 224)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))