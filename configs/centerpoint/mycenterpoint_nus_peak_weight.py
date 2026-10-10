_base_ = ['./centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

train_cfg = dict(max_epochs=20, val_interval=2)

data_root = '/home/bjtc/data/datasets/nuScenes/'

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl'))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl'))

model = dict(
    train_cfg=dict(
        pts=dict(
            evidence_radius=dict(
                enabled=True,
                mode='peak_weight',

                # center-peak weighting（仅 mode='peak_weight' 生效）
                peak_weight_lambda=0.25,
                peak_weight_max=1.5,

                # global_cls_id 顺序：
                # 0 car
                # 1 truck
                # 2 construction_vehicle
                # 3 bus
                # 4 trailer
                # 5 barrier
                # 6 motorcycle
                # 7 bicycle
                # 8 pedestrian
                # 9 traffic_cone

                mu_count=[
                    4.600432050998271,   # car
                    5.541817288432705,   # truck
                    5.334663480738094,   # construction_vehicle
                    6.419398504239150,   # bus
                    5.989245405647676,   # trailer
                    4.701634883567867,   # barrier
                    4.160940715201492,   # motorcycle
                    3.775921268516510,   # bicycle
                    3.442334732439508,   # pedestrian
                    3.269097472751511,   # traffic_cone
                ],

                std_count=[
                    2.222942567231153,   # car
                    2.205420851636537,   # truck
                    1.934533612976618,   # construction_vehicle
                    2.031991294418419,   # bus
                    2.166347479482141,   # trailer
                    1.874690625198825,   # barrier
                    1.823096042518524,   # motorcycle
                    1.488910639160438,   # bicycle
                    1.328912324266508,   # pedestrian
                    1.442800876873920,   # traffic_cone
                ],

                mu_density=[
                    2.281077402024760,   # car
                    2.223644560238401,   # truck
                    2.021749004247342,   # construction_vehicle
                    2.141615324575096,   # bus
                    1.881765385094933,   # trailer
                    4.607991331727368,   # barrier
                    3.416348788955743,   # motorcycle
                    3.593107509007616,   # bicycle
                    3.643917409550010,   # pedestrian
                    5.027951886916142,   # traffic_cone
                ],

                std_density=[
                    1.783494957172735,   # car
                    1.657444411800424,   # truck
                    1.539421729131514,   # construction_vehicle
                    1.590335150326655,   # bus
                    1.469092037663843,   # trailer
                    1.886044657567828,   # barrier
                    1.761683147265199,   # motorcycle
                    1.526143570251275,   # bicycle
                    1.336600874405485,   # pedestrian
                    1.617468697898159,   # traffic_cone
                ],

                beta=0.5,
                alpha=0.25,
                sparse_score_max=2.0,
                max_radius=15,
            )
        )
    )
)
