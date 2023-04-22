_base_ = [
    '../_base_/models/erfnet_fcn.py', '../_base_/datasets/crack_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

model = dict(
    decode_head=dict(num_classes=2))
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