# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.95, min_lr=1e-6, by_epoch=True)  # by_epoch改为True
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)   #type 从 IterBasedRunner 改为 EpochBasedRunner   /  max_iters 改为 max_epochs
checkpoint_config = dict(by_epoch=True, interval=10)   # interval表示经过多少次iter后保存一次模型。 by_epoch改为True后, interval表示为epoch数
evaluation = dict(interval=10, metric=['mIoU', 'mDice', 'mFscore'], save_best='mIoU', pre_eval=True)  # interval表示经过多少次iter后评估一次模型 ，但是当 type='EpochBasedRunner', interval表示的就是epoch数。
