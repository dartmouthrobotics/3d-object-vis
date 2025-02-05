import numpy as np
import pickle
import matplotlib.image as Image
import copy

# MJ from to kitti script in asv_3d_perception
# ----------------------------------------------------------

CLASS_REMAP = {"Car": "ship", "Pedestrian": "buoy", "Cyclist": "other"}

def box3d_kitti_camera_to_lidar(box3d_camera, calib):
    """
    # unified normative format 

    Args:
        box3d_camera: one box info  [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        box3d_lidar: one box info [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center (unified normative format )

    """
    box3d_camera_copy = copy.deepcopy(box3d_camera)
    xyz_camera, r = box3d_camera_copy[0:3], box3d_camera_copy[6]
    l, h, w = box3d_camera_copy[3], box3d_camera_copy[4], box3d_camera_copy[5] 

    xyz_camera_np =np.array(xyz_camera).reshape(-1, np.shape(xyz_camera)[0]) # (1,3)

    # print("xyz camera_np {} and shape {}".format(xyz_camera_np, np.shape(xyz_camera_np)))
    xyz_lidar = calib.rect_to_lidar(xyz_camera_np) # (1,3)
    # print("xyz_lidar {} and shape {}".format(xyz_lidar, np.shape(xyz_lidar)))
    xyz_lidar = xyz_lidar[0] # (1,3) --> len(3) array
    xyz_lidar[2] += h / 2 # unified normative format 
    # print("xyz lidar {}".format(xyz_lidar))
    # print("l {} w {} h {} r {}".format(l, w, h, r))
    return [xyz_lidar[0], xyz_lidar[1], xyz_lidar[2], l, w, h, -(r + np.pi / 2)] # unified normative format 



def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    # unified normative format 

    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center (unified normative format )

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6] 

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2 # unified normative format 
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1) # unified normative format 

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[0].strip().split(" ")[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(" ")[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(" ")[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {
        "P2": P2.reshape(3, 4),
        "R0": R0.reshape(3, 3),
        "Tr_velo2cam": Tr_velo_to_cam.reshape(3, 4),
    }


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
            # print(calib)
        else:
            calib = calib_file

        self.P2 = calib["P2"]  # 3 x 4
        self.R0 = calib["R0"]  # 3 x 3
        self.V2C = calib["Tr_velo2cam"]  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1
        )
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2
        )  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate(
            (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1
        )
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner
# ----------------------------------------------------------



class SceneDataset:
    def __init__(self, data_root, result_file, gt_info_file=None, second_format=False):
        """
        Args:
            - data_root: kitti data dir
                - velodyne
                - image_2
            - result_file: result.pkl path
            - gt_info_file: kitti_infos_val.pkl path
        """
        self.data_root = data_root
        self.result_file = result_file
        self.points_path = data_root  / 'velodyne'
        self.image_path = data_root  / 'image_2'

        self.frame_id = None
        self.second_format = second_format # MJ --> determine which __getitem__ function to be used

        self.pred_list = self.get_pickle(self.result_file)
        self.gt_info_file = gt_info_file
        # 是否有 gt 选框，一般为 kitti_infos_val.pkl
        if gt_info_file is not None: 
            self.val_list = self.get_pickle(self.gt_info_file)
            print("len pred {}".format(len(self.pred_list))) # MJ
            print("len val_list {}".format(len(self.val_list))) # MJ
            # print("pred data {}".format(self.pred_list[0]))
            # print("gt data {}".format(self.val_list[0]))

            # AQL
            # if len(self.pred_list) < len(self.val_list): 
            if len(self.pred_list) <= len(self.val_list): # MJ exist still emtpy possible (same length)
                filtered_val_list = []
                filtered_pred_list = []
                for p in self.pred_list:
                    
                    add = False
                    for g in self.val_list:
                        if not self.second_format:
                            if p["frame_id"] == g["image"]["image_idx"]:
                                add = True
                        else:
                            if len(p["image_idx"]) > 0:
                                if p["image_idx"][0] == g["image_idx"]:
                                    add = True
                            else:
                                break
                        if add:
                            filtered_val_list.append(g)
                            break
                    if add:
                        filtered_pred_list.append(p)

                # for e in self.val_list: # Assuming they follow the same order
                    # add = False
                    # if not self.second_format:
                    #     if self.pred_list[len(filtered_val_list)]["frame_id"] == e["image"]["image_idx"]:
                    #         add = True
                    # else:
                    #     if self.pred_list[len(filtered_val_list)]["image_idx"][0] == e["image_idx"]:
                    #         add = True
                    # if add:
                    #     print(len(filtered_val_list))
                    #     filtered_val_list.append(e)
                
                self.pred_list = filtered_pred_list
                self.val_list = filtered_val_list
            # AQL END


            assert len(self.pred_list) == len(self.val_list), "pred & val don't match"
            self.iter_len = len(self.val_list) # MJ


    def __len__(self):
        return len(self.pred_list)
    
    def __getitem__(self, idx:int):
        """
        Return idx sample's batch_dict:
            - points
            - pred_boxes
            - pred_name
            - gt_boxes
            - gt_name
            - calib
            - image
        """
        if not self.second_format:
            print("THIS IS OPENPCDET FORMAT: index {}".format(idx))
            # 获得样本 id
            data_dict = {}
            frame_id = self.pred_list[idx]['frame_id']
            if self.gt_info_file is not None:
                val_id = self.val_list[idx]['point_cloud']['lidar_idx']
                assert frame_id == val_id, f'pred_id: {frame_id}, val_id: {val_id} not the same'
                
                self.frame_id = frame_id # MJ

                # 获得 gt_boxes 及其 name: score 标注
                data_dict['gt_boxes'] = self.val_list[idx]['annos']['gt_boxes_lidar']
                num_boxes = data_dict['gt_boxes'].shape[0]
                name = self.val_list[idx]['annos']['name'][:num_boxes]
                score = [1.0 for i in range(num_boxes)]
                name = self.change_name(name) # MJ
                data_dict['gt_name'] = self.name_with_score(name, score)

                # 获得 calib matrix
                data_dict['calib'] = dict(
                    Tr_velo_to_cam = self.val_list[idx]['calib']['Tr_velo_to_cam'],
                    P2 = self.val_list[idx]['calib']['P2'],
                    R0_rect = self.val_list[idx]['calib']['R0_rect']
                )

            # 获得 points & pred_boxes
            velo_file = self.points_path / f'{frame_id}.bin'
            # img_file = self.image_path / f'{frame_id}.png'
            img_file = self.image_path / f'{frame_id}.jpg' # MJ
            points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
            data_dict['points'] = points
            data_dict['pred_boxes'] = self.pred_list[idx]['boxes_lidar']

            # 获得 name: score 标注
            score = self.pred_list[idx]['score']
            name = self.pred_list[idx]['name']
            name = self.change_name(name) # MJ remap
            data_dict['pred_name'] = self.name_with_score(name, score)

            data_dict['image'] = Image.imread(str(img_file))
            return data_dict

        else: # SECOND format
            print("THIS IS SECOND FORMAT: index {}".format(idx))
            data_dict = {}
            try:
                frame_id = self.pred_list[idx]['frame_id']
            except: # For second.pytorch1.5 base
                # print(self.pred_list[idx])
                frame_id = self.pred_list[idx]['image_idx'][0] # array
                frame_id = f"{frame_id:06d}"
            if self.gt_info_file is not None:
                # GT information
                try:
                    val_id = self.val_list[idx]['point_cloud']['lidar_idx']
                except: # For second.pytorch1.5 base
                    val_id = self.val_list[idx]['image_idx'] # int
                    val_id = f"{val_id:06d}"
                assert frame_id == val_id, f'pred_id: {frame_id}, val_id: {val_id} not the same'

                self.frame_id = frame_id
                # print("HERE")
                # ---------------------------------------
                # MJ gt_box --> camera (kitti) --> lidar

                #### 1) process camera coordinate-based pkl information -> lidar coordinate-based information
                # 获得 gt_boxes 及其 name: score 标注
                # data_dict['gt_boxes'] = self.val_list[idx]['annos']['gt_boxes_lidar'] 
                # idx = 0
                location = self.val_list[idx]['annos']['location']
                dimensions = self.val_list[idx]['annos']['dimensions'] # lhw (originally hwl in label.txt -> they converted for DB pkl)
                rotation_y = self.val_list[idx]['annos']['rotation_y']
                # Combine into a list of lists
                boxes_3d_camera = [[*loc.tolist(), *dim.tolist(), rot] for loc, dim, rot in zip(location, dimensions, rotation_y)]

                # print(boxes_3d_camera)
                # camera coordinate processor
                # print("HERE2")
                # 2) calibration info read as non-homogeneous
                calib_tmp = dict(
                    Tr_velo2cam = self.val_list[idx]['calib/Tr_velo_to_cam'][:3,],
                    P2 = self.val_list[idx]['calib/P2'][:3,],
                    R0 = self.val_list[idx]['calib/R0_rect'][:3,:3]
                )
                # print("calib {}".format(calib_tmp))

                # 3) gt_box --> camera-based --> lidar-based
                calib_object = Calibration(calib_tmp)
                boxes_3d_lidar = []
                for box_3d_camera in boxes_3d_camera:
                    box_3d_lidar = box3d_kitti_camera_to_lidar(box_3d_camera, calib_object)
                    boxes_3d_lidar.append(box_3d_lidar)
                # print("boxes 3d lidar {}".format(boxes_3d_lidar))
                data_dict['gt_boxes'] = np.array(boxes_3d_lidar) # list -> np.array (like OpenPCDet)

                # print("HERE3")

                # ---------------------------------------
                # as it is OK
                num_boxes = data_dict['gt_boxes'].shape[0]
                name = self.val_list[idx]['annos']['name'][:num_boxes] # ok
                score = [1.0 for i in range(num_boxes)]
                name = self.change_name(name) # MJ
                data_dict['gt_name'] = self.name_with_score(name, score) # ok

                # 获得 calib matrix
                # data_dict['calib'] = dict(
                #     Tr_velo_to_cam = self.val_list[idx]['calib']['Tr_velo_to_cam'],
                #     P2 = self.val_list[idx]['calib']['P2'],
                #     R0_rect = self.val_list[idx]['calib']['R0_rect']
                # )

                # as it is OK
                data_dict['calib'] = dict(
                    Tr_velo_to_cam = self.val_list[idx]['calib/Tr_velo_to_cam'],
                    P2 = self.val_list[idx]['calib/P2'],
                    R0_rect = self.val_list[idx]['calib/R0_rect']
                )

            # ---------------------------------------
            # Use as it is
            # 获得 points & pred_boxes
            velo_file = self.points_path / f'{frame_id}.bin'
            # img_file = self.image_path / f'{frame_id}.png'
            img_file = self.image_path / f'{frame_id}.jpg' # MJ
            points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
            data_dict['points'] = points # ok

            # ---------------------------------------
            # MJ pred_box --> camera-based --> lidar-based    
            # data_dict['pred_boxes'] = self.pred_list[idx]['boxes_lidar'] # CHANGE

            #### 1) process camera coordinate-based pkl information -> lidar coordinate-based information
            location = self.pred_list[idx]['location']
            dimensions = self.pred_list[idx]['dimensions'] # lhw (originally hwl in label.txt -> they converted for DB pkl)
            rotation_y = self.pred_list[idx]['rotation_y']
            # Combine into a list of lists
            boxes_3d_camera = [[*loc.tolist(), *dim.tolist(), rot] for loc, dim, rot in zip(location, dimensions, rotation_y)]

            # camera coordinate processor
            # 2) calibration info read as non-homogeneous
            calib_tmp = dict(
                Tr_velo2cam = self.val_list[idx]['calib/Tr_velo_to_cam'][:3,],
                P2 = self.val_list[idx]['calib/P2'][:3,],
                R0 = self.val_list[idx]['calib/R0_rect'][:3,:3]
            )
            # print("calib {}".format(calib_tmp))
            # print("HERE6")

            # 3) pred_box --> camera-based --> lidar-based
            calib_object = Calibration(calib_tmp)
            boxes_3d_lidar = []
            for box_3d_camera in boxes_3d_camera:
                box_3d_lidar = box3d_kitti_camera_to_lidar(box_3d_camera, calib_object)
                boxes_3d_lidar.append(box_3d_lidar)
            # print("boxes 3d lidar {}".format(boxes_3d_lidar))
            # print("HERE7")
            data_dict['pred_boxes'] = np.array(boxes_3d_lidar) # list -> np.array (like OpenPCDet)

            # ---------------------------------------
            # Use as it is
            # 获得 name: score 标注
            score = self.pred_list[idx]['score'] # ok
            name = self.pred_list[idx]['name']# ok
            name = self.change_name(name) # MJ
            data_dict['pred_name'] = self.name_with_score(name, score)

            data_dict['image'] = Image.imread(str(img_file)) # ok
            return data_dict

    def change_name(self, name_list):
        new_name_list = []
        for name in name_list:
            new_name = CLASS_REMAP.get(name)
            new_name_list.append(new_name)
        return new_name_list

    def __getitemtest__(self, idx:int):
        """
        Return idx sample's batch_dict:
            - points
            - pred_boxes
            - pred_name
            - gt_boxes
            - gt_name
            - calib
            - image
        
        # MJ observation
        SECOND-based pkl: image-based 
        OpenPCDet-based pkl: lidar-based 
        """
        # 获得样本 id
        data_dict = {}
        try:
            frame_id = self.pred_list[idx]['frame_id']
        except: # For second.pytorch1.5 base
            frame_id = self.pred_list[idx]['image_idx'][0] # array
            frame_id = f"{frame_id:06d}"
        if self.gt_info_file is not None:
            # GT information
            try:
                val_id = self.val_list[idx]['point_cloud']['lidar_idx']
            except: # For second.pytorch1.5 base
                val_id = self.val_list[idx]['image_idx'] # int
                val_id = f"{val_id:06d}"
            assert frame_id == val_id, f'pred_id: {frame_id}, val_id: {val_id} not the same'

            # print("HERE")
            # ---------------------------------------
            # MJ gt_box --> camera (kitti) --> lidar

            #### 1) process camera coordinate-based pkl information -> lidar coordinate-based information
            # 获得 gt_boxes 及其 name: score 标注
            # data_dict['gt_boxes'] = self.val_list[idx]['annos']['gt_boxes_lidar'] 
            # idx = 0
            location = self.val_list[idx]['annos']['location']
            dimensions = self.val_list[idx]['annos']['dimensions'] # lhw (originally hwl in label.txt -> they converted for DB pkl)
            rotation_y = self.val_list[idx]['annos']['rotation_y']
            # Combine into a list of lists
            boxes_3d_camera = [[*loc.tolist(), *dim.tolist(), rot] for loc, dim, rot in zip(location, dimensions, rotation_y)]

            # print(boxes_3d_camera)
            # camera coordinate processor
            # print("HERE2")
            # 2) calibration info read as non-homogeneous
            calib_tmp = dict(
                Tr_velo2cam = self.val_list[idx]['calib/Tr_velo_to_cam'][:3,],
                P2 = self.val_list[idx]['calib/P2'][:3,],
                R0 = self.val_list[idx]['calib/R0_rect'][:3,:3]
            )
            # print("calib {}".format(calib_tmp))

            # 3) gt_box --> camera-based --> lidar-based
            calib_object = Calibration(calib_tmp)
            boxes_3d_lidar = []
            for box_3d_camera in boxes_3d_camera:
                box_3d_lidar = box3d_kitti_camera_to_lidar(box_3d_camera, calib_object)
                boxes_3d_lidar.append(box_3d_lidar)
            # print("boxes 3d lidar {}".format(boxes_3d_lidar))
            data_dict['gt_boxes'] = np.array(boxes_3d_lidar) # list -> np.array (like OpenPCDet)

            # print("HERE3")

            # ---------------------------------------
            # as it is OK
            num_boxes = data_dict['gt_boxes'].shape[0]
            name = self.val_list[idx]['annos']['name'][:num_boxes] # ok
            score = [1.0 for i in range(num_boxes)]
            data_dict['gt_name'] = self.name_with_score(name, score) # ok

            # 获得 calib matrix
            # data_dict['calib'] = dict(
            #     Tr_velo_to_cam = self.val_list[idx]['calib']['Tr_velo_to_cam'],
            #     P2 = self.val_list[idx]['calib']['P2'],
            #     R0_rect = self.val_list[idx]['calib']['R0_rect']
            # )

            # as it is OK
            data_dict['calib'] = dict(
                Tr_velo_to_cam = self.val_list[idx]['calib/Tr_velo_to_cam'],
                P2 = self.val_list[idx]['calib/P2'],
                R0_rect = self.val_list[idx]['calib/R0_rect']
            )

        # ---------------------------------------
        # Use as it is
        # 获得 points & pred_boxes
        velo_file = self.points_path / f'{frame_id}.bin'
        # img_file = self.image_path / f'{frame_id}.png'
        img_file = self.image_path / f'{frame_id}.jpg' # MJ
        points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        data_dict['points'] = points # ok

        # ---------------------------------------
        # MJ pred_box --> camera-based --> lidar-based    
        # data_dict['pred_boxes'] = self.pred_list[idx]['boxes_lidar'] # CHANGE

        #### 1) process camera coordinate-based pkl information -> lidar coordinate-based information
        location = self.pred_list[idx]['location']
        dimensions = self.pred_list[idx]['dimensions'] # lhw (originally hwl in label.txt -> they converted for DB pkl)
        rotation_y = self.pred_list[idx]['rotation_y']
        # Combine into a list of lists
        boxes_3d_camera = [[*loc.tolist(), *dim.tolist(), rot] for loc, dim, rot in zip(location, dimensions, rotation_y)]

        # camera coordinate processor
        # 2) calibration info read as non-homogeneous
        calib_tmp = dict(
            Tr_velo2cam = self.val_list[idx]['calib/Tr_velo_to_cam'][:3,],
            P2 = self.val_list[idx]['calib/P2'][:3,],
            R0 = self.val_list[idx]['calib/R0_rect'][:3,:3]
        )
        # print("calib {}".format(calib_tmp))
        # print("HERE6")

        # 3) pred_box --> camera-based --> lidar-based
        calib_object = Calibration(calib_tmp)
        boxes_3d_lidar = []
        for box_3d_camera in boxes_3d_camera:
            box_3d_lidar = box3d_kitti_camera_to_lidar(box_3d_camera, calib_object)
            boxes_3d_lidar.append(box_3d_lidar)
        # print("boxes 3d lidar {}".format(boxes_3d_lidar))
        # print("HERE7")
        data_dict['pred_boxes'] = np.array(boxes_3d_lidar) # list -> np.array (like OpenPCDet)

        # ---------------------------------------
        # Use as it is
        # 获得 name: score 标注
        score = self.pred_list[idx]['score'] # ok
        name = self.pred_list[idx]['name']# ok
        print("name", name)
        name = CLASS_REMAP.get(name) # MJ
        data_dict['pred_name'] = self.name_with_score(name, score)

        data_dict['image'] = Image.imread(str(img_file)) # ok
        return data_dict

    @staticmethod
    def get_pickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def name_with_score(name, score):
        """
        将名字和得分以 name: score 的形式，放到一个 array 当中
        """
        assert len(name) == len(score), "name & score don't match"
        ret_list = []
        for i in range(len(name)):
            s = f'{name[i]}: {score[i]:.2f}'
            ret_list.append(s)
        ret_array = np.array(ret_list)
        return ret_array

if __name__ == '__main__':
    from pathlib import Path
    data_root = Path(__file__).parent / 'kitti_data'
    dataset = SceneDataset(data_root)
    for i in range(10):
        dataset[i]