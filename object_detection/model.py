import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from 
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def boxes_to_corners_2d(boxes3d):
    """
    Args:
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
        corners_2d: (N, 4, 2) corners in bird's eye view
    """
    
    corners = np.zeros((boxes3d.shape[0], 4, 2))
    
    
    for i in range(boxes3d.shape[0]):
        x, y, z, dx, dy, dz, heading = boxes3d[i]
        
        
        corner_x = np.array([dx/2, dx/2, -dx/2, -dx/2]) 
        corner_y = np.array([dy/2, -dy/2, -dy/2, dy/2])
        
        
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading), np.cos(heading)]])
        corners_rotated = np.dot(R, np.stack([corner_x, corner_y]))
        
        
        corners[i, :, 0] = corners_rotated[0] + x
        corners[i, :, 1] = corners_rotated[1] + y
    
    return corners


def visualize_bev(points, boxes=None, scores=None, labels=None, class_names=None, 
                 xlim=(-50, 50), ylim=(-50, 50), figsize=(10, 10), save_path=None):
    """
    Visualize bird's eye view of points and 3D boxes
    Args:
        points: (N, 3+) [x, y, z, ...], point cloud
        boxes: (M, 7) [x, y, z, dx, dy, dz, heading], 3D boxes
        scores: (M,) confidence scores
        labels: (M,) class labels
        class_names: list of class names
        xlim, ylim: plot ranges
        figsize: figure size
        save_path: path to save the figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c='black', alpha=0.5)
    
    
    if boxes is not None:
        corners_2d = boxes_to_corners_2d(boxes)
        
        
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        
        for i, corners in enumerate(corners_2d):
            
            color_idx = labels[i] if labels is not None else 0
            color = colors[color_idx % len(colors)]
            
            
            for k in range(4):
                j = (k + 1) % 4
                ax.plot([corners[k, 0], corners[j, 0]], 
                        [corners[k, 1], corners[j, 1]], 
                        color=color, linewidth=2)
                
            
            front_middle = (corners[0] + corners[1]) / 2
            center = (corners[0] + corners[1] + corners[2] + corners[3]) / 4
            ax.plot([center[0], front_middle[0]], 
                    [center[1], front_middle[1]], 
                    color=color, linewidth=3)
            
            
            if class_names is not None and labels is not None:
                label_text = class_names[labels[i]]
                if scores is not None:
                    label_text += f': {scores[i]:.2f}'
                ax.text(center[0], center[1], label_text, 
                        fontsize=8, color='white', 
                        bbox=dict(facecolor=color, alpha=0.5))
    
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Bird\'s Eye View')
    
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"BEV visualization saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    return fig


class PCDetector:
    def __init__(self, 
                 cfg_file='cfgs/kitti_models/second.yaml', 
                 data_path='dataset/', 
                 ckpt='models/second.pth', 
                 ext='.bin', 
                 save_dir='visualization_results'):
        """
        A class for 3D object detection using PCDet.
        
        Args:
            cfg_file: Path to the config file
            data_path: Path to the point cloud data
            ckpt: Path to the checkpoint file
            ext: File extension of point cloud files (.bin or .npy)
            save_dir: Directory to save visualization results
        """
        self.cfg_file = cfg_file
        self.data_path = Path(data_path)
        self.ckpt = ckpt
        self.ext = ext
        self.save_dir = save_dir
        
        # Load config
        cfg_from_yaml_file(self.cfg_file, cfg)
        self.cfg = cfg
        
        # Create logger
        self.logger = common_utils.create_logger()
        
        # Create save directory if needed
        if self.save_dir:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(exist_ok=True, parents=True)
            
        # Setup dataset and model
        self._setup_model()
        
    def _setup_model(self):
        """Initialize dataset and model"""
        self.demo_dataset = DemoDataset(
            dataset_cfg=self.cfg.DATA_CONFIG, 
            class_names=self.cfg.CLASS_NAMES, 
            training=False,
            root_path=self.data_path, 
            ext=self.ext, 
            logger=self.logger
        )
        
        self.logger.info(f'Total number of samples: \t{len(self.demo_dataset)}')
        
        self.model = build_network(
            model_cfg=self.cfg.MODEL, 
            num_class=len(self.cfg.CLASS_NAMES), 
            dataset=self.demo_dataset
        )
        self.model.load_params_from_file(filename=self.ckpt, logger=self.logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()
    
    def detect(self, idx=0, visualize=True):
        """
        Perform detection on a specific sample index
        
        Args:
            idx: Sample index to process
            visualize: Whether to visualize the results
            
        Returns:
            dict: Detection results containing boxes, scores, and labels
        """
        if idx >= len(self.demo_dataset):
            self.logger.error(f"Index {idx} out of range. Dataset has {len(self.demo_dataset)} samples.")
            return None
            
        self.logger.info(f'Processing sample index: \t{idx + 1}')
        data_dict = self.demo_dataset[idx]
        data_dict = self.demo_dataset.collate_batch([data_dict])
        
        with torch.no_grad():
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
        
        points = data_dict['points'][:, 1:4].cpu().numpy()
        boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        labels = pred_dicts[0]['pred_labels'].cpu().numpy()
        
        # Log detection results
        self.logger.info(f"Detection Results: {len(boxes)} objects found")
        for i, box in enumerate(boxes):
            if scores[i] > 0.5:
                class_name = self.cfg.CLASS_NAMES[labels[i]-1 if labels[i] > 0 else 0]
                self.logger.info(f"Object {i}: {class_name}, Score={scores[i]:.2f}, Box={box.tolist()}")
        
        # Visualization
        save_path = None
        if visualize:
            if self.save_dir:
                save_path = self.save_dir / f'bev_{idx:03d}.png'
            
            visualize_bev(
                points=points, 
                boxes=boxes, 
                scores=scores, 
                labels=labels, 
                class_names=self.cfg.CLASS_NAMES,
                xlim=(-50, 50),
                ylim=(-50, 50),
                save_path=save_path
            )
        
        return {
            'points': points,
            'pred_boxes': boxes,
            'pred_scores': scores,
            'pred_labels': labels,
            'class_names': self.cfg.CLASS_NAMES
        }
    
    def detect_all(self, visualize=True):
        """
        Perform detection on all samples in the dataset
        
        Args:
            visualize: Whether to visualize the results
            
        Returns:
            list: List of detection results for all samples
        """
        results = []
        for idx in range(len(self.demo_dataset)):
            result = self.detect(idx, visualize)
            results.append(result)
        
        self.logger.info('Completed processing all samples.')
        return results


"""Legacy function to maintain backward compatibility"""
detector = PCDetector(
    cfg_file='cfgs/kitti_models/second.yaml',
    data_path='dataset/',
    ckpt='models/second.pth',
    ext='.bin',
    save_dir='visualization_results'
)

detector.logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
detector.detect_all(visualize=True)
detector.logger.info('Completed.')

