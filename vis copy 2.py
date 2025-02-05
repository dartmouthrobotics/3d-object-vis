from viewer.viewer import Viewer
from scenedataset import SceneDataset
from tqdm import tqdm
from utils import *
import os

# def build_viewer(box_type="OpenPCDet", bg=(255,255,255), offscreen=False, remote=False):
def build_viewer(box_type="OpenPCDet", bg=(255,255,255), offscreen=False, remote=False):

    # in case you are working on a remote machine, specify DISPLAY value like this
    # checkout https://github.com/marcomusy/vedo/issues/64  
    if remote: os.environ['DISPLAY'] = ':99.0'
    return Viewer(box_type=box_type, bg=bg, offscreen=offscreen)

def kitti_visualization(dataset: SceneDataset, class_list, vis_num, thres = None, 
                        save_path: Path=None, all_iter=False):
    """
    Draw kitti scene and inference results
    Args:
        - dataset: SceneDataset
        - class_list: classes need to show
        - vis_num: number of frames to show
        - thres: object score under thres will not show
        - save_path: directory to save the visualization results
                     if save, the visualization won't show on screen
    """
    offscreen = False
    if save_path is not None:
        offscreen = True
        # save_img_path = save_path / 'img'
        # save_velo_path = save_path / 'velo'
        save_img_path = os.path.join(save_path,'img') # MJ
        save_velo_path = os.path.join(save_path,'velo') # MJ

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_img_path, exist_ok=True)
        os.makedirs(save_velo_path, exist_ok=True)

        # save_path.mkdir(parents=True, exist_ok=True)
        # save_img_path.mkdir(parents=True, exist_ok=True)
        # save_velo_path.mkdir(parents=True, exist_ok=True)

    # TODO total frame number for GT length
    ##### TMP
    vis_num = dataset.iter_len 
    # vis_num = dataset.iter_len if all_iter else vis_num

    pbar = tqdm(range(vis_num)) 
    pbar.set_description('visualize')
    # https://stackoverflow.com/questions/1409886/help-maximum-number-of-clients-reached-segmentation-fault
    # vi = build_viewer(offscreen=offscreen) # MJ inside for loop -> max client
    for idx in pbar:
        vi = build_viewer(offscreen=offscreen) #### TMP

        data = dataset[idx]
        points = data['points']


        vi.add_points(points[:,0:3],
                    radius = 3,
                    color= 'dimgray',
                    # scatter_filed = points[:,2],
                    alpha=1,
                    del_after_show = True,
                    add_to_3D_scene = True,
                    add_to_2D_scene = False)
                    # color_map_name = 'Greys')

        if dataset.result_file:
            boxes = data['pred_boxes']
            box_info = data['pred_name']
            # 是否设置阈值筛选
            if thres is not None:
                thres_mask = [float(i.split(':')[1]) > thres for i in box_info]
                boxes = boxes[thres_mask]
                box_info = box_info[thres_mask]

            # add pred_boxes
            vi.add_3D_boxes(boxes=boxes[:,0:7],
                            box_info=box_info,
                            color="green",
                            mesh_alpha = 0.1,  # 表面透明度
                            show_corner_spheres = True,    # 是否展示顶点上的球
                            caption_size=(0.1,0.1),
                            add_to_2D_scene=True,
                            )

        # 是否有 gt 信息
        if dataset.gt_info_file is not None:
            gt_boxes = data['gt_boxes']
            gt_name = data['gt_name']
            class_mask = [i.split(':')[0] in class_list for i in gt_name]
            gt_boxes = gt_boxes[class_mask,:]
            gt_name = gt_name[class_mask]

            # add gt_boxes
            vi.add_3D_boxes(gt_boxes,
                            color='red',
                            box_info=gt_name,
                            caption_size=(0.1,0.1),
                            is_label=True
                            )
                            
        # 如果使用 kitti raw 数据集则需要将 calib.txt 放到 data_root 下
        calib_file = dataset.data_root / 'calib.txt'
        if calib_file.exists():
            data['calib'] = get_calib(calib_file)

        if data.get('calib', None) is not None:
            # set calib matrix
            V2C = data['calib']['Tr_velo_to_cam']
            P2 = data['calib']['P2']
            R0_rect = data['calib']['R0_rect']

            vi.set_extrinsic_mat(V2C)
            vi.set_intrinsic_mat(P2)
            vi.set_r0_rect_mat(R0_rect)

            image = data['image']
            vi.add_image(image)
        """
        save_name = save_img_path / f'{idx:0>4d}.png' if save_path is not None else None
        vi.show_2D(save_name=save_name)
        save_name = save_velo_path / f'{idx:0>4d}.png' if save_path is not None else None
        vi.show_3D(save_name=save_name)
        """

        # MJ
        save_frame_id = dataset.frame_id # frame number extract
        print("save frame id {}".format(save_frame_id))

        open_numbers = [3315, 3318, 3320, 3331, 3336, 3373, 3396, 3406, 3407, 3408]
        # Convert to 6-digit strings
        opensetsix_digit_strings = [str(num).zfill(6) for num in open_numbers]


        # if save_frame_id is not None:
        if save_frame_id is not None and save_frame_id in opensetsix_digit_strings:
            # save_name = os.path.join(save_img_path,f'{save_frame_id}.png')  if save_path is not None else None
            # vi.show_2D(save_name=save_name) # MJ
            save_name = os.path.join(save_velo_path,f'{save_frame_id}.png') if save_path is not None else None
            vi.show_3D(save_name=save_name)

    if offscreen:
        print(f'visualization results are saved to: {save_path}')

def seg_vis(velo_root: Path, results, save_path: Path=None):
    """
    Draw segmentation scene
    Args:
        - velo_root: point cloud data path
        - results: inference result path, i.e. result.pkl path
        - save_path: path where you want to save your visualization results
                     if save, the visualization won't show on screen
    """
    sample_number = len(results)
    pbar = tqdm(range(sample_number))
    # pbar = tqdm(range(5))
    for idx in pbar:
        frame_id = results[idx]['frame_id']
        velo_file = velo_root / (frame_id + '.bin')
        points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        points_seg = results[idx]['points_seg']

        offscreen = False
        if save_path is not None:
            offscreen = True
            save_path.mkdir(exist_ok=True)

        vi = build_viewer(offscreen=offscreen)
        vi.add_points(points[:,:3], alpha=0.3)
        if len(points_seg) > 0:
            vi.add_points(points_seg, scatter_filed=points_seg[:,2])
        # vi.add_origin()

        save_name = save_path / f'{idx:0>4d}.png' if save_path is not None else None
        vi.show_3D(save_name=save_name)
    if offscreen:
        print(f'visualization results are saved to: {save_path}')

"""
if __name__ == "__main__":
    vi = build_viewer()
    i= 0
    pseudo_points = np.random.randn(100, 3) # your points
    pseudo_boxes = np.array([[i*0.05, -1, 1, 1, 1, 1, 0], [i*0.05, 1, 1, 1, 1, 1, 0]]) # your boxes

    vi.add_points(pseudo_points)   # (N, 3), (x, y, z)
    vi.add_3D_boxes(pseudo_boxes)  # (N, 7), (x, y, z, w, h, l, theta)
    vi.show_3D()
"""

if __name__ == '__main__':
    # kitti scene script

    data_root = Path('data_root')
    ALL_ITER = False # True -> auto saving in output folder
    VAL_DATA = False # False -> test data

    # ----------------------
    # SECOND Family
    # method = "second"
    method = "CLOCs"

    # ----------------------
    # OpenPCDet Family
    # method="voxel_rcnn"
    # method = "focal_conv_f"
    # method = "pointpainting"
    # method = "pointpillars"
    # method = "pv_rcnn"
    # method = "pointrcnn"
    # ----------------------

    methods = ["CLOCs", 
            #    "voxel_rcnn", 
            #    "focal_conv_f", 
            #    "pointpainting", 
            #    "pointpillars", 
            #    "pv_rcnn", 
            #    "pointrcnn"
               ]
    

    for method in methods:

        # output_folder = os.path.join("output", method) if ALL_ITER else None
        output_folder = os.path.join("output_final", method) if ALL_ITER else None
        if output_folder is not None:
            output_folder = os.path.join(output_folder, "val-data") if VAL_DATA else os.path.join(output_folder, "test-data")
        # ----------------------
        result_folder = os.path.join("result_folder", method)
        result_folder = os.path.join(result_folder, "val-data") if VAL_DATA else os.path.join(result_folder, "test-data")
        result_file = Path(os.path.join(result_folder, 'result.pkl'))
        gt_info_file = Path(os.path.join(result_folder, 'kitti_infos_val.pkl'))


        # different argument pass for __getitem__ definition
        if method  == "second" or method == "CLOCs":
            dataset = SceneDataset(data_root, result_file, gt_info_file, second_format=True)
        else:
            dataset = SceneDataset(data_root, result_file, gt_info_file)


        # class_list = ['Car', 'Pedestrian', 'Cyclist'] 
        # class_list = ['Car']
        class_list = ['ship'] # for gt_name to appear at drawing MJ
        kitti_visualization(dataset, 
                    class_list,
                    vis_num=100, 
                    thres = 0.49, save_path=output_folder, all_iter=ALL_ITER)

