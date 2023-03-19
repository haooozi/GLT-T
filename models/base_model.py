""" 
baseModel.py
Created by zenn at 2021/5/9 14:40
"""
import matplotlib as mp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision
from utils.metrics import estimateOverlap, estimateAccuracy
import torch.nn.functional as F
import numpy as np
import copy
import os
import open3d as o3d
import math
from datasets.data_classes import Box
from pyquaternion import Quaternion
from utils.LineMesh import LineMesh
from nuscenes.utils import geometry_utils


class BaseModel(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()
        self.frames = 0


    def plot_open3d_score(self, sequence, frame_id, pred_bbox, seed_points, scores, capture_path):
        this_bb = sequence[frame_id]["3d_bbox"]
        this_pc = sequence[frame_id]["pc"]
        view_pc = points_utils.cropAndCenterPC(this_pc, this_bb, offset=2)[0]  # offset control the size of scene
        view_pc = view_pc.points[:, ::1]
        view_pc = view_pc.swapaxes(0, 1)

        seed_points_pc = seed_points

        view_bb = Box([0, 0, 0], this_bb.wlh, Quaternion())
        pred_bbox.translate(-this_bb.center)
        pred_bbox.rotate(this_bb.orientation.inverse)

        view_bb_corners = view_bb.corners()
        pred_bbox_corners = pred_bbox.corners()
        gt_bbox = view_bb_corners.swapaxes(0, 1)
        pred_bbox = pred_bbox_corners.swapaxes(0, 1)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=400, height=400)
        render_option = vis.get_render_option()
        render_option.point_size = 5

        point_cloud_scene = o3d.geometry.PointCloud()
        point_cloud_seed_points = o3d.geometry.PointCloud()
        point_cloud_scene.points = o3d.utility.Vector3dVector(view_pc)
        point_cloud_seed_points.points = o3d.utility.Vector3dVector(seed_points_pc)

        # scene_color = np.ones((view_pc.shape[0], 3)) * 0.2
        # dists = point_cloud_scene.compute_point_cloud_distance(point_cloud_seed_points)
        # dists = np.asarray(dists)
        # index = np.where(dists < 0.01)[0]
        # for i in index:
        #     scene_color[i] = [1., 0., 0.]
        # point_cloud_scene.colors = o3d.utility.Vector3dVector(scene_color)


        scores = scores.detach().cpu().numpy()[0]
        scores = (scores-scores.min())/(scores.max()-scores.min())
        seed_color = np.ones((seed_points_pc.shape[0], 3))
        for i in range(seed_points_pc.shape[0]):
            score = scores[i]
            if score <= 0.125:
                seed_color[i] = seed_color[i] * [0., 0., 143 + max(0, math.ceil(score / 0.015625 - 1)) * 16] #(0,0,143)->(0,0,255)
            elif score <= 0.375:
                seed_color[i] = seed_color[i] * [0., min(16 * ((score - 0.125) / 0.015625),255) / 255, 1.] #(0,0,255)->(0,255,255)
            elif score <= 0.625:
                seed_color[i] = seed_color[i] * [min(((score - 0.375) / 0.015625) * 16, 255) / 255,
                                                                   1.0, (255 - min(((score - 0.375) / 0.015625) * 16, 255))/255]#(0,255,255)->(255,255,0)
            elif score <= 0.875:
                seed_color[i] = seed_color[i] * [1., (255 - min(((score - 0.625) / 0.015625) * 16, 255)) / 255, 0.] #(255,255,0)->(255,0,0)
            else:
                seed_color[i] = seed_color[i] * [(255 - min(((score - 0.875) / 0.015625) * 16, 255)) / 255, 0., 0.] #(255,0,0)->(128,0,0)
        point_cloud_seed_points.colors = o3d.utility.Vector3dVector(seed_color)

        scene_color = np.ones((view_pc.shape[0], 3))
        dists = np.sum((view_pc[:, None] - seed_points_pc[None, :]) ** 2, axis=-1)
        knn_idx = dists.argsort()[:, 0]
        for i in range(view_pc.shape[0]):
            scene_color[i] = seed_color[knn_idx[i]]

        point_cloud_scene.colors = o3d.utility.Vector3dVector(scene_color)

        # bbox
        lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                              [4, 5], [5, 6], [6, 7], [7, 4],
                              [0, 4], [1, 5], [2, 6], [3, 7]])  # (12,2)

        gt_colors = np.array([[0., 1., 0.] for _ in range(len(lines_box))])  # green
        gt_line_mesh = LineMesh(gt_bbox, lines_box, gt_colors, radius=0.03)

        pred_colors = np.array([[1., 0., 0.] for _ in range(len(lines_box))])  # red
        pred_line_mesh = LineMesh(pred_bbox, lines_box, pred_colors, radius=0.03)

        vis.add_geometry(point_cloud_scene)
        # vis.add_geometry(point_cloud_seed_points)
        gt_line_mesh.add_line(vis)
        pred_line_mesh.add_line(vis)

        if frame_id == 1:
            vis.run()
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)
            if not capture_path is None:
                vis.capture_screen_image(capture_path)
            vis.destroy_window()
        else:
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('view.json')
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.run()
            # save picture
            if not capture_path is None:
                vis.capture_screen_image(capture_path)
            vis.destroy_window()

    def plot_open3d_line(self, sequence, frame_id, pred_bbox, search_pc, search_xyz, vote_xyz, capture_path):

        this_bb = sequence[frame_id]["3d_bbox"]
        this_pc = sequence[frame_id]["pc"]
        view_pc = points_utils.cropAndCenterPC(this_pc, this_bb, offset=2)[0]  # offset control the size of scene
        view_pc = view_pc.points[:, ::2]
        view_pc = view_pc.swapaxes(0, 1)

        view_bb = Box([0, 0, 0], this_bb.wlh, Quaternion())
        view_bb_corners = view_bb.corners()
        gt_bbox = view_bb_corners.swapaxes(0, 1)

        # view_bb_center = view_bb.center
        # gt_center = view_bb_center[:,None].swapaxes(0, 1)


        pred_bbox.translate(-this_bb.center)
        pred_bbox.rotate(this_bb.orientation.inverse)
        pred_bbox_corners = pred_bbox.corners()
        pred_bbox = pred_bbox_corners.swapaxes(0, 1)

        lines_offset = np.concatenate((search_xyz, vote_xyz), axis=0)
        # print(lines_offset.shape)

        sample_points = view_pc #search_pc
        seed_points = search_xyz
        proposals = vote_xyz

        dist = np.linalg.norm(proposals,ord=2,axis=1)
        index = np.where(dist < 1.5)[0]

        proposals = proposals[index]

        lines_offset = np.concatenate((seed_points[index], proposals), axis=0)


        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=800, height=800)
        render_option = vis.get_render_option()
        render_option.point_size = 3

        point_cloud_sample_points = o3d.geometry.PointCloud()
        point_cloud_seed_points = o3d.geometry.PointCloud()
        point_cloud_proposals = o3d.geometry.PointCloud()
        point_cloud_sample_points.points = o3d.utility.Vector3dVector(sample_points)
        point_cloud_seed_points.points = o3d.utility.Vector3dVector(seed_points)
        point_cloud_proposals.points = o3d.utility.Vector3dVector(proposals)

        sample_points_color = np.ones((sample_points.shape[0], 3)) * 0.2
        seed_points_color = np.ones((seed_points.shape[0], 3)) * [1., 0., 0.]
        proposals_color = np.ones((proposals.shape[0], 3)) * [1., 0., 0.]

        """-----score-----"""
        # proposals_color = np.ones((proposals.shape[0], 3))
        # scores = dist[index]
        # scores = 1.5 - scores
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        # for i in range(proposals.shape[0]):
        #     score = scores[i]
        #     if score <= 0.125:
        #         proposals_color[i] = proposals_color[i] * [0., 0., 143 + max(0, math.ceil(score / 0.015625 - 1)) * 16] #(0,0,143)->(0,0,255)
        #     elif score <= 0.375:
        #         proposals_color[i] = proposals_color[i] * [0., min(16 * ((score - 0.125) / 0.015625),255) / 255, 1.] #(0,0,255)->(0,255,255)
        #     elif score <= 0.625:
        #         proposals_color[i] = proposals_color[i] * [min(((score - 0.375) / 0.015625) * 16, 255) / 255,
        #                                                            1.0, (255 - min(((score - 0.375) / 0.015625) * 16, 255))/255]#(0,255,255)->(255,255,0)
        #     elif score <= 0.875:
        #         proposals_color[i] = proposals_color[i] * [1., (255 - min(((score - 0.625) / 0.015625) * 16, 255)) / 255, 0.] #(255,255,0)->(255,0,0)
        #     else:
        #         proposals_color[i] = proposals_color[i] * [(255 - min(((score - 0.875) / 0.015625) * 16, 255)) / 255, 0., 0.] #(255,0,0)->(128,0,0)

        # point_cloud_sample_points.colors = o3d.utility.Vector3dVector(sample_points_color)
        # point_cloud_seed_points.colors = o3d.utility.Vector3dVector(seed_points_color)
        point_cloud_proposals.colors = o3d.utility.Vector3dVector(proposals_color)

        # dists = point_cloud_sample_points.compute_point_cloud_distance(point_cloud_seed_points)
        # dists = np.asarray(dists)
        # index = np.where(dists < 0.01)[0]
        # for i in index:
        #     sample_points_color[i] = [1., 0., 0.]
        point_cloud_sample_points.colors = o3d.utility.Vector3dVector(sample_points_color)

        # bbox
        lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                              [4, 5], [5, 6], [6, 7], [7, 4],
                              [0, 4], [1, 5], [2, 6], [3, 7]])  # (12,2)

        gt_colors = np.array([[0., 1., 0.] for _ in range(len(lines_box))])  # red
        gt_line_mesh = LineMesh(gt_bbox, lines_box, gt_colors, radius=0.01)
        pred_colors = np.array([[0., 1., 0.] for _ in range(len(lines_box))])
        pred_line_mesh = LineMesh(pred_bbox, lines_box, pred_colors, radius=0.01)

        num = proposals.shape[0]
        lines = np.ones((num, 2), dtype=np.int64)  # (128,2)
        for i in range(num):
            lines[i, 0] = i
            lines[i, 1] = i + num
        # line_colors = np.array([[255/255*0.2, 255/255*0.2, 0/255*0.2] for _ in range(len(lines))])
        line_colors = np.array([[0., 1., 0.] for _ in range(len(lines))])
        line_line_mesh = LineMesh(lines_offset, lines, line_colors, radius=0.0075)


        vis.add_geometry(point_cloud_sample_points)
        # vis.add_geometry(point_cloud_seed_points)
        vis.add_geometry(point_cloud_proposals)

        # gt_line_mesh.add_line(vis)
        line_line_mesh.add_line(vis)
        # pred_line_mesh.add_line(vis)

        if frame_id == 1:
            vis.run()
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)

            if not capture_path is None:
                vis.capture_screen_image(capture_path)

            vis.destroy_window()
        else:
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('view.json')
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.run()

            # save picture
            if not capture_path is None:
                vis.capture_screen_image(capture_path)

            vis.destroy_window()

    def plot_open3d_bbox(self, sequence, frame_id, pred_bbox, capture_path):
        this_bb = sequence[frame_id]["3d_bbox"]
        this_pc = sequence[frame_id]["pc"]
        view_pc = points_utils.cropAndCenterPC(this_pc, this_bb, offset=10)[0]  # offset control the size of scene
        pc_gt = points_utils.cropAndCenterPC(this_pc, this_bb, offset=0)[0]

        view_pc = view_pc.points[:, ::1]
        view_pc = view_pc.swapaxes(0, 1)
        pc_gt = pc_gt.points[:, ::1]
        pc_gt = pc_gt.swapaxes(0, 1)

        view_bb = Box([0, 0, 0], this_bb.wlh, Quaternion())
        pred_bbox.translate(-this_bb.center)
        pred_bbox.rotate(this_bb.orientation.inverse)

        view_bb_corners = view_bb.corners()
        pred_bbox_corners = pred_bbox.corners()
        gt_bbox = view_bb_corners.swapaxes(0, 1)
        pred_bbox = pred_bbox_corners.swapaxes(0, 1)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=800, height=800)
        render_option = vis.get_render_option()
        render_option.point_size = 2

        point_cloud_scene = o3d.geometry.PointCloud()
        point_cloud_gt = o3d.geometry.PointCloud()
        point_cloud_scene.points = o3d.utility.Vector3dVector(view_pc)
        point_cloud_gt.points = o3d.utility.Vector3dVector(pc_gt)

        scene_color = np.ones((view_pc.shape[0], 3)) * 0.2
        # dists = point_cloud_scene.compute_point_cloud_distance(point_cloud_gt)
        # dists = np.asarray(dists)
        # index = np.where(dists < 0.01)[0]
        # for i in index:
        #     scene_color[i] = [1., 0., 0.]
        point_cloud_scene.colors = o3d.utility.Vector3dVector(scene_color)
        gt_color = np.array([[1., 0., 0.] for _ in range(pc_gt.shape[0])])
        # gt_color = np.ones((pc_gt.shape[0], 3)) * 0.6
        point_cloud_gt.colors = o3d.utility.Vector3dVector(gt_color)
        # bbox
        lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                              [4, 5], [5, 6], [6, 7], [7, 4],
                              [0, 4], [1, 5], [2, 6], [3, 7]])  # (12,2)

        gt_colors = np.array([[1., 0., 0.] for _ in range(len(lines_box))])  # green
        gt_line_mesh = LineMesh(gt_bbox, lines_box, gt_colors, radius=0.03)

        pred_colors = np.array([[0., 1., 0.] for _ in range(len(lines_box))])  # red
        pred_line_mesh = LineMesh(pred_bbox, lines_box, pred_colors, radius=0.03)

        vis.add_geometry(point_cloud_scene)
        # vis.add_geometry(point_cloud_gt)
        gt_line_mesh.add_line(vis)
        pred_line_mesh.add_line(vis)

        if frame_id == 1:
            vis.run()
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)
            if not capture_path is None:
                vis.capture_screen_image(capture_path)
            vis.destroy_window()
        else:
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('view.json')
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.run()
            # save picture
            if not capture_path is None:
                vis.capture_screen_image(capture_path)
            vis.destroy_window()

    def plot_open3d_scene(self, sequence, frame_id, pred_bbox, capture_path):
        this_bb = sequence[frame_id]["3d_bbox"]
        this_pc = sequence[frame_id]["pc"]
        pc_gt = points_utils.cropAndCenterPC(this_pc, this_bb, offset=0)[0]

        this_pc = this_pc.points[:,::3]
        this_pc = this_pc.swapaxes(0, 1)
        pc_gt = pc_gt.points[:,::3]
        pc_gt = pc_gt.swapaxes(0, 1)

        view_bb = Box([0, 0, 0], this_bb.wlh, Quaternion())
        pred_bbox.translate(-this_bb.center)
        pred_bbox.rotate(this_bb.orientation.inverse)

        view_bb_corners = view_bb.corners()
        pred_bbox_corners = pred_bbox.corners()
        gt_bbox = view_bb_corners.swapaxes(0, 1)
        pred_bbox = pred_bbox_corners.swapaxes(0, 1)


        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=800, height=800)
        render_option = vis.get_render_option()
        render_option.point_size = 3


        point_cloud_scene = o3d.geometry.PointCloud()
        point_cloud_gt = o3d.geometry.PointCloud()
        point_cloud_scene.points = o3d.utility.Vector3dVector(this_pc)
        point_cloud_gt.points = o3d.utility.Vector3dVector(pc_gt)

        scene_color = np.ones((this_pc.shape[0], 3)) * 0.6
        # dists = point_cloud_scene.compute_point_cloud_distance(point_cloud_gt)
        # dists = np.asarray(dists)
        # index = np.where(dists < 0.01)[0]
        # for i in index:
        #     scene_color[i] = [1., 0., 0.]
        point_cloud_scene.colors = o3d.utility.Vector3dVector(scene_color)
        gt_color = np.array([[1., 0., 0.] for _ in range(pc_gt.shape[0])])
        point_cloud_gt.colors = o3d.utility.Vector3dVector(gt_color)
        # bbox
        lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                              [4, 5], [5, 6], [6, 7], [7, 4],
                              [0, 4], [1, 5], [2, 6], [3, 7]])  #(12,2)

        gt_colors = np.array([[0., 1., 0.] for _ in range(len(lines_box))])  # green
        gt_line_mesh = LineMesh(gt_bbox, lines_box, gt_colors, radius=0.03)

        pred_colors = np.array([[1., 0., 0.] for _ in range(len(lines_box))])  # red
        pred_line_mesh = LineMesh(pred_bbox, lines_box, pred_colors, radius=0.03)


        vis.add_geometry(point_cloud_scene)
        vis.add_geometry(point_cloud_gt)
        gt_line_mesh.add_line(vis)
        pred_line_mesh.add_line(vis)

        if frame_id == 1:
            vis.run()
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)
            if not capture_path is None:
                vis.capture_screen_image(capture_path)
            vis.destroy_window()
        else:
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('view.json')
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.run()
            # save picture
            if not capture_path is None:
                vis.capture_screen_image(capture_path)
            vis.destroy_window()


    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, data, output):
        raise NotImplementedError

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        raise NotImplementedError

    def evaluate_one_sample(self, data_dict, ref_box):
        end_points = self(data_dict)

        estimation_box = end_points['estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)
        return candidate_box, end_points, best_box_idx

    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(sequence)):  # tracklet
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
            else:

                # construct input dict
                data_dict, ref_bb = self.build_input_dict(sequence, frame_id, results_bbs)
                # run the tracker
                candidate_box, end_points, best_box_idx = self.evaluate_one_sample(data_dict, ref_box=ref_bb)
                results_bbs.append(candidate_box)

                """--------------Visualization---------------"""

                # pre_bb = copy.deepcopy(candidate_box)

                # """--attention/scores--"""
                # search_xyz = end_points['search_xyz'].cpu().numpy()[0]  # seed points
                # importances = end_points['importances'].sigmoid()
                # os.makedirs(
                #     os.path.join("visualizations", "score"),
                #     exist_ok=True)
                # capture_path = os.path.join("visualizations", "score", f"{frame_id}.png")
                # self.plot_open3d_score(sequence, frame_id=frame_id, pred_bbox=pre_bb, seed_points=search_xyz ,scores=importances, capture_path=capture_path)

                # """--offset lines--"""
                # search_xyz = end_points['search_xyz'].cpu().numpy()[0] #seed points
                # vote_xyz = end_points['vote_xyz'].cpu().numpy()[0]
                # search_c = data_dict['search_points'].cpu().numpy()[0] #sampled points
                # os.makedirs(
                #     os.path.join("visualizations", "line"),
                #     exist_ok=True)
                # capture_path = os.path.join("visualizations", "line", f"{frame_id}.png")
                # self.plot_open3d_line(sequence, frame_id=frame_id, pred_bbox=pre_bb, search_pc=search_c, search_xyz=search_xyz, vote_xyz=vote_xyz,
                #                         capture_path=capture_path)

                # """--bbox--"""
                # os.makedirs(
                #     os.path.join("visualizations", "bbox"),
                #     exist_ok=True)
                # capture_path = os.path.join("visualizations", "bbox", f"{frame_id}.png")
                # self.plot_open3d_bbox(sequence, frame_id=frame_id, pred_bbox=pre_bb, capture_path=capture_path)

                # """--scene+bbox--"""
                # os.makedirs(
                #     os.path.join("visualizations", "scene"),
                #     exist_ok=True)
                # capture_path = os.path.join("visualizations", "scene", f"{frame_id}.jpg")
                # self.plot_open3d_scene(sequence, frame_id=frame_id, pred_bbox=pre_bb, capture_path=capture_path)

                """-------------------------------------------"""

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances, results_bbs

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1

        """-------------split points number on first frames------------"""
        # first_pc = sequence[0]['pc']
        # first_bb = sequence[0]['3d_bbox']
        # first_pc_gt = points_utils.cropAndCenterPC(first_pc, first_bb, offset=0)[0]
        # if first_pc_gt.points.shape[1] >= 10 and first_pc_gt.points.shape[1] < 50:  #threshold
        #     # print(first_pc_gt.points.shape[1])
        #     self.frames = self.frames + len(sequence)
        #     # print(self.frames)
        #     ious, distances = self.evaluate_one_sequence(sequence)
        #     # update metrics
        #     self.success(torch.tensor(ious, device=self.device))
        #     self.prec(torch.tensor(distances, device=self.device))
        #     self.log('success/test', self.success, on_step=True, on_epoch=True)
        #     self.log('precision/test', self.prec, on_step=True, on_epoch=True)
        """-------------------------------------------------------------"""

        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, result_bbs = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)
        return result_bbs

    def test_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, pos=1, alpha=0.3, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.pos = pos
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-7

    def forward(self, inputs, targets, mask=None): #(2,128) (2,128)
        if mask == None:
            inputs = inputs.view(-1) #(256)
            targets = targets.view(-1) #(256)
            pred = torch.sigmoid(inputs) #(256)

            pos = targets.data.eq(1).nonzero().squeeze().cuda()  #---------index
            neg = targets.data.eq(0).nonzero().squeeze().cuda()

            pred_pos = torch.index_select(pred, 0, pos)
            targets_pos = torch.index_select(targets, 0, pos)

            pred_neg = torch.index_select(pred, 0, neg)
            targets_neg = torch.index_select(targets, 0, neg)

            loss_pos = -1 * torch.pow((1-pred_pos),self.gamma) * torch.log(pred_pos + self.eps) * self.pos

            loss_neg = -1 * torch.pow((pred_neg),self.gamma) * torch.log(1 - pred_neg + self.eps)

            loss = torch.cat((loss_pos, loss_neg), dim=0)

            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss

            return loss
        else:
            inputs = inputs.view(-1)  # (256)
            targets = targets.view(-1)  # (256)
            pred = torch.sigmoid(inputs)  # (256)
            mask = mask.view(-1)

            pos = targets.data.eq(1).nonzero().squeeze().cuda()  # ---------index
            neg = targets.data.eq(0).nonzero().squeeze().cuda()

            pred_pos = torch.index_select(pred, 0, pos)
            targets_pos = torch.index_select(targets, 0, pos)
            mask_pos = torch.index_select(mask, 0, pos)

            pred_neg = torch.index_select(pred, 0, neg)
            targets_neg = torch.index_select(targets, 0, neg)
            mask_neg = torch.index_select(mask, 0, neg)

            loss_pos = -1 * torch.pow((1 - pred_pos), self.gamma) * torch.log(pred_pos + self.eps) * self.pos * mask_pos

            loss_neg = -1 * torch.pow((pred_neg), self.gamma) * torch.log(1 - pred_neg + self.eps) * mask_neg

            loss = torch.cat((loss_pos, loss_neg), dim=0)

            if self.size_average:
                loss = torch.sum(loss) / (torch.sum(mask) + 1e-6)
            else:
                loss = loss

            return loss

criterion_imp = FocalLoss(pos=1, gamma=2, size_average=True).cuda()
criterion_objective = FocalLoss(pos=2, gamma=2, size_average=True).cuda()

class MatchingBaseModel(BaseModel):

    def compute_loss(self, data, output):
        """
        :param data: input data
        :param output:
        :return:
        """
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        seg_label = data['seg_label']
        box_label = data['box_label']  # B,4
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]
        importances = output['importances']

        maxi_half = data['maxi_half']
        mini_half = data['mini_half']


        loss_importances = criterion_imp(importances, seg_label)
        loss_vote = F.smooth_l1_loss(vote_xyz, box_label[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
        loss_vote = (loss_vote.mean(2) * seg_label * (1 + importances.sigmoid())).sum() / (seg_label.sum() + 1e-06)

        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        objectness_label[dist < 0.3] = 1

        objectness_score = estimation_boxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1

        loss_objective = criterion_objective(objectness_score, objectness_label, objectness_mask)

        loss_box = F.smooth_l1_loss(estimation_boxes[:, :, :4],
                                    box_label[:, None, :4].expand_as(estimation_boxes[:, :, :4]),
                                    reduction='none')

        loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        return {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_importances,
            "loss_vote": loss_vote,
        }

    def generate_template(self, sequence, current_frame_id, results_bbs):
        """
        generate template for evaluating.
        the template can be updated using the previous predictions.
        :param sequence: the list of the whole sequence
        :param current_frame_id:
        :param results_bbs: predicted box for previous frames
        :return:
        """
        first_pc = sequence[0]['pc']
        previous_pc = sequence[current_frame_id - 1]['pc']
        if "firstandprevious".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([first_pc, previous_pc],
                                                               [results_bbs[0], results_bbs[current_frame_id - 1]],
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        elif "first".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(first_pc, results_bbs[0],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "previous".upper() in self.config.hape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(previous_pc, results_bbs[current_frame_id - 1],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "all".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([frame["pc"] for frame in sequence[:current_frame_id]],
                                                               results_bbs,
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        return template_pc, canonical_box

    def generate_search_area(self, sequence, current_frame_id, results_bbs):
        """
        generate search area for evaluating.

        :param sequence:
        :param current_frame_id:
        :param results_bbs:
        :return:
        """
        this_bb = sequence[current_frame_id]["3d_bbox"]
        this_pc = sequence[current_frame_id]["pc"]
        if ("previous_result".upper() in self.config.reference_BB.upper()):
            ref_bb = results_bbs[-1]
        elif ("previous_gt".upper() in self.config.reference_BB.upper()):
            previous_bb = sequence[current_frame_id - 1]["3d_bbox"]
            ref_bb = previous_bb
        elif ("current_gt".upper() in self.config.reference_BB.upper()):
            ref_bb = this_bb
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset)
        return search_pc_crop, ref_bb

    def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):
        """
        construct input dict for evaluating
        :param template_pc:
        :param search_pc:
        :param template_box:
        :return:
        """
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)
        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
        }
        return data_dict

    def build_input_dict(self, sequence, frame_id, results_bbs, **kwargs):
        # preparing search area
        search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)
        # update template
        template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)
        # construct input dict
        data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box)
        return data_dict, ref_bb


class MotionBaseModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]
        prev_pc = prev_frame['pc']
        this_pc = this_frame['pc']
        ref_box = results_bbs[-1]
        prev_frame_pc = points_utils.generate_subwindow(prev_pc, ref_box,
                                                        scale=self.config.bb_scale,
                                                            offset=self.config.bb_offset)
        this_frame_pc = points_utils.generate_subwindow(this_pc, ref_box,
                                                        scale=self.config.bb_scale,
                                                        offset=self.config.bb_offset)

        canonical_box = points_utils.transform_box(ref_box, ref_box)
        prev_points, idx_prev = points_utils.regularize_pc(prev_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1)

        this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T,
                                                           self.config.point_sample_size,
                                                           seed=1)
        seg_mask_prev = geometry_utils.points_in_box(canonical_box, prev_points.T, 1.25).astype(float)

        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        if frame_id != 1:
            seg_mask_prev[seg_mask_prev == 0] = 0.2
            seg_mask_prev[seg_mask_prev == 1] = 0.8
        seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

        timestamp_prev = np.full((self.config.point_sample_size, 1), fill_value=0)
        timestamp_this = np.full((self.config.point_sample_size, 1), fill_value=0.1)
        prev_points = np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
        this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

        stack_points = np.concatenate([prev_points, this_points], axis=0)

        data_dict = {"points": torch.tensor(stack_points[None, :], device=self.device, dtype=torch.float32),
                     }
        if getattr(self.config, 'box_aware', False):
            candidate_bc_prev = points_utils.get_point_to_box_distance(
                stack_points[:self.config.point_sample_size, :3], canonical_box)
            candidate_bc_this = np.zeros_like(candidate_bc_prev)
            candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)
            data_dict.update({'candidate_bc': points_utils.np_to_torch_tensor(candidate_bc.astype('float32'),
                                                                              device=self.device)})
        return data_dict, results_bbs[-1]
