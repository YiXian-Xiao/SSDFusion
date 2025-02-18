checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean='%mean%',
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std='%std%',
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = '%dataroot%'
dataset_type = 'BaseSegDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=16000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
ham_norm_cfg = dict(num_groups=32, requires_grad=True, type='GN')
img_norm_cfg = dict(
    mean='%mean%', std='%std%', to_rgb=True)
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
metainfo = dict(
    classes=(
        'unlabelled',
        'car',
        'person',
        'bike',
        'curve',
        'car_stop',
        'guardrail',
        'color_cone',
        'bump',
    ),
    pallete=[
        [
            0,
            0,
            0,
        ],
        [
            64,
            0,
            128,
        ],
        [
            64,
            64,
            0,
        ],
        [
            0,
            128,
            192,
        ],
        [
            0,
            0,
            192,
        ],
        [
            128,
            128,
            0,
        ],
        [
            64,
            64,
            128,
        ],
        [
            192,
            128,
            128,
        ],
        [
            192,
            64,
            0,
        ],
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attention_kernel_paddings=[
            2,
            [
                0,
                3,
            ],
            [
                0,
                5,
            ],
            [
                0,
                10,
            ],
        ],
        attention_kernel_sizes=[
            5,
            [
                1,
                7,
            ],
            [
                1,
                11,
            ],
            [
                1,
                21,
            ],
        ],
        depths=[
            3,
            3,
            5,
            2,
        ],
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=[
            32,
            64,
            160,
            256,
        ],
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth',
            type='Pretrained'),
        mlp_ratios=[
            8,
            8,
            4,
            4,
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        type='MSCAN'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean='%mean%',
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std='%std%',
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        ham_channels=256,
        ham_kwargs=dict(
            MD_R=16,
            MD_S=1,
            eval_steps=7,
            inv_t=100,
            rand_init=True,
            train_steps=6),
        in_channels=[
            64,
            160,
            256,
        ],
        in_index=[
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=9,
        type='LightHamHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=160000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='test/fused', seg_map_path='test/Segmentation_labels'),
        data_root='%dataroot%',
        img_suffix='.png',
        metainfo=dict(
            classes=(
                'unlabelled',
                'car',
                'person',
                'bike',
                'curve',
                'car_stop',
                'guardrail',
                'color_cone',
                'bump',
            ),
            pallete=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    64,
                    0,
                    128,
                ],
                [
                    64,
                    64,
                    0,
                ],
                [
                    0,
                    128,
                    192,
                ],
                [
                    0,
                    0,
                    192,
                ],
                [
                    128,
                    128,
                    0,
                ],
                [
                    64,
                    64,
                    128,
                ],
                [
                    192,
                    128,
                    128,
                ],
                [
                    192,
                    64,
                    0,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                480,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='BaseSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        480,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=16000)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='train/fused', seg_map_path='train/Segmentation_labels'),
        data_root='%dataroot%',
        img_suffix='.png',
        metainfo=dict(
            classes=(
                'unlabelled',
                'car',
                'person',
                'bike',
                'curve',
                'car_stop',
                'guardrail',
                'color_cone',
                'bump',
            ),
            pallete=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    64,
                    0,
                    128,
                ],
                [
                    64,
                    64,
                    0,
                ],
                [
                    0,
                    128,
                    192,
                ],
                [
                    0,
                    0,
                    192,
                ],
                [
                    128,
                    128,
                    0,
                ],
                [
                    64,
                    64,
                    128,
                ],
                [
                    192,
                    128,
                    128,
                ],
                [
                    192,
                    64,
                    0,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='BaseSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='test/fused', seg_map_path='test/Segmentation_labels'),
        data_root='%dataroot%',
        img_suffix='.png',
        metainfo=dict(
            classes=(
                'unlabelled',
                'car',
                'person',
                'bike',
                'curve',
                'car_stop',
                'guardrail',
                'color_cone',
                'bump',
            ),
            pallete=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    64,
                    0,
                    128,
                ],
                [
                    64,
                    64,
                    0,
                ],
                [
                    0,
                    128,
                    192,
                ],
                [
                    0,
                    0,
                    192,
                ],
                [
                    128,
                    128,
                    0,
                ],
                [
                    64,
                    64,
                    128,
                ],
                [
                    192,
                    128,
                    128,
                ],
                [
                    192,
                    64,
                    0,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                480,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='BaseSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/segnext_mscan-t_1xb16-adamw-160k_msrs-512x512'
