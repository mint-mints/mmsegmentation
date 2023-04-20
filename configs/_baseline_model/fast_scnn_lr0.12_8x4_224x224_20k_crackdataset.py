_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/crack_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

# Re-config the data sampler.
data = dict(samples_per_gpu=4, workers_per_gpu=4)

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=4e-5)

model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
crop_size = (224, 224)