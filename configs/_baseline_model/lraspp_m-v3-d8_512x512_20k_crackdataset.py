_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py', '../_base_/datasets/crack_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

model = dict(pretrained='open-mmlab://contrib/mobilenet_v3_large',
             decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2)
             )

# Re-config the data sampler.
data = dict(samples_per_gpu=4, workers_per_gpu=4)

runner = dict(type='IterBasedRunner', max_iters=320000)

crop_size = (512, 512)
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