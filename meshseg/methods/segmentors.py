import torch
import trimesh
import numpy as np
import scipy.optimize
from tqdm.auto import tqdm
import potpourri3d as pp3d

from copy import deepcopy
from collections import Counter
from collections import defaultdict
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import cv2

from ..models.GLIP.glip import GLIPModel
from ..models.SAM.sam import SAMModel

class BaseMeshSegmentor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rendered_images = None
        self.rendered_images_face_ids = None
        self.prompts = None
        self.mesh = None

    def color_mesh_and_save(self, face_cls, colors, output_filename="test.obj"):
        # Now color the facees according to the predicted class
        face_colors = torch.zeros((len(self.mesh.faces), 4), dtype=torch.uint8)

        for i in range(len(self.mesh.faces)):
            face_colors[i][0:3] = torch.tensor(colors[face_cls[i]]) * 255
            face_colors[i][-1] = 255

        # Create trimesh
        scene = trimesh.Scene()
        output_mesh = trimesh.Trimesh(
            vertices=self.mesh.vertices.cpu().numpy(),
            faces=self.mesh.faces.cpu().numpy(),
        )
        output_mesh.visual.face_colors = face_colors.cpu().numpy()
        scene.add_geometry(output_mesh, node_name="output")
        _ = scene.export(output_filename)

class BaseDetMeshSegmentor(BaseMeshSegmentor):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.bbox_predictions = None

    def set_mesh(self, mesh):
        self.mesh = mesh

    def set_prompts(self, prompts):
        self.prompts = prompts

    def set_rendered_views(
        self, rendered_images: torch.Tensor, images_face_ids: torch.Tensor
    ):
        self.rendered_images = rendered_images
        self.rendered_images_face_ids = images_face_ids

    def get_included_face_ids(self, pixel_face_ids, bbox_cor, face_counter):
        relevant_face_ids = defaultdict(int)
        non_relevant_face_ids = defaultdict(int)

        bbox_cor = bbox_cor.to(torch.int64)
        (col_min, row_min), (col_max, row_max) = (
            bbox_cor[:2].tolist(),
            bbox_cor[2:].tolist(),
        )

        for col in range(col_min, col_max):
            for row in range(row_min, row_max):
                face_id = pixel_face_ids[row][col].item()
                if face_id == -1:
                    continue
                relevant_face_ids[face_id] += 1
        
        # my_face_counter = pixel_face_ids[row_min:row_max, col_min:col_max].cpu().numpy()
        # my_face_counter = my_face_counter[my_face_counter != -1]
        # my_relevant_face_ids = dict(Counter(list(my_face_counter)))

        for k, _ in face_counter.items():
            if k == -1:
                continue
            if k not in relevant_face_ids:
                non_relevant_face_ids[k] += 1

        return relevant_face_ids, non_relevant_face_ids

    def get_included_face_ids_from_mask(self, pixel_face_ids, mask, bbox, face_counter):
        relevant_face_ids = defaultdict(int)
        non_relevant_face_ids = defaultdict(int)

        bbox = bbox.to(torch.int64)
        (col_min, row_min), (col_max, row_max) = (
            bbox[:2].tolist(),
            bbox[2:].tolist(),
        )

        for col in range(col_min, col_max):
            for row in range(row_min, row_max):
                if (mask[row][col]):
                    face_id = pixel_face_ids[row][col].item()
                    if face_id == -1:
                        continue
                    relevant_face_ids[face_id] += 1

        for k, _ in face_counter.items():
            if k == -1:
                continue
            if k not in relevant_face_ids:
                non_relevant_face_ids[k] += 1

        return relevant_face_ids, non_relevant_face_ids
        
    # def process_box_predictions(self, prompt, preds, rendering_face_ids):
    #     # Initialize a score vector for each face in the mesh for the given region prompt (e.g. "the leg of a person").
    #     face_view_prompt_score = np.zeros((len(self.mesh.faces)))
    #     face_view_freq = np.zeros((len(self.mesh.faces)))

    #     pred_bboxes = preds[0][1]
    #     pred_bboxes_cor = preds[0][1].bbox
    #     n_boxes = len(
    #         pred_bboxes
    #     )  # The number of predcited bounding boxes for the given prompt (e.g., the leg of a person).

    #     face_counter = rendering_face_ids.flatten().cpu().int().numpy()
    #     face_counter = face_counter[face_counter != -1]
    #     face_counter = Counter(list(face_counter))

    #     for i in range(n_boxes):
    #         if pred_bboxes.get_field("labels")[i].item() != 1 and "of" in prompt:
    #             continue

    #         confidence_score = pred_bboxes.get_field("scores")[i].item()

    #         # Get the included face ids inside this bounding box
    #         included_face_ids, not_included_face_ids = self.get_included_face_ids(
    #             rendering_face_ids, pred_bboxes_cor[i], face_counter
    #         )

    #         for k, v in included_face_ids.items():
    #             face_view_prompt_score[k] += v * confidence_score

    #     return face_view_prompt_score, face_view_freq

    def __call__(self):
        pass

class GLIPSAMMeshSegmenter(BaseDetMeshSegmentor):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.glip_model = GLIPModel()
        self.sam_model = SAMModel()

        self.colors_dict = {
            0: [255, 0, 0],   # Red
            1: [0, 255, 0],   # Green
            2: [0, 0, 255],   # Blue
            3: [255, 255, 0],   # Yellow
            4: [255, 0, 255],   # Magenta
            5: [0, 255, 255],   # Cyan
            6: [128, 0, 0], # Dark Red
            7: [0, 128, 0], # Dark Green
            8: [0, 0, 128], # Dark Blue
            9: [128, 128, 128] # Gray
        }

    def predict_face_cls(self):
        # Get the bounding boxes predictions for the given prompts
        self.predict_bboxes()
        assert self.bbox_predictions is not None
        print(f"Finished GLIP")

        if self.cfg.satr.sam:
            self.predict_exact_masks()
            print('Finished SAM')

        face_cls = np.zeros((len(self.mesh.faces), len(self.prompts)))
        face_freq = np.zeros((len(self.mesh.faces), len(self.prompts)))

        # Looping over the views
        for i, view in tqdm(enumerate(self.rendered_images)):
            for j, prompt in enumerate(self.prompts):
                print(f'Processing view: {i}, Prompt: {j}')
                if self.cfg.satr.sam:
                    (
                        face_view_prompt_score,
                        face_view_prompt_freq,
                    ) = self.process_masks_predictions(
                        prompt,
                        self.masks_predictions[i][prompt],
                        self.bbox_predictions[i][prompt][0][1],
                        self.rendered_images_face_ids[i].squeeze(0),
                    )
                else:
                    (
                        face_view_prompt_score,
                        face_view_prompt_freq,
                    ) = self.process_box_predictions(
                        prompt,
                        self.bbox_predictions[i][prompt],
                        self.rendered_images_face_ids[i].squeeze(0),
                    )

                face_cls[:, j] += face_view_prompt_score
                face_freq[:, j] += face_view_prompt_freq

        return face_cls, face_freq

    def predict_bboxes(self):
        # Generate GLIP predictions for every rendered image
        print("Feeding the views to GLIP...")
        self.bbox_predictions = []
        num_views = len(self.rendered_images)

        # fig, axs = plt.subplots(2,5,figsize=(80,22))
        fig, axs = plt.subplots(3,4,figsize=(30,22))
        # fig, axs = plt.subplots(1,2,figsize=(80,22))

        for i in range(len(self.rendered_images)):
            self.bbox_predictions.append({})

            # img = self.rendered_images[i].permute([1, 2, 0]) * 256.0
            img = self.rendered_images[i].permute([1, 2, 0]) * 255.0
            img = img.to(torch.uint8)
            img_to_show = img.cpu().numpy().copy()
            ax = axs[i // 4, i % 4]
            # ax = axs[i]
            for p_id, p in enumerate(self.prompts):
                print('GLIP - View:', i, 'Prompt:', p_id, end=' ')
                
                res = self.glip_model.predict(img.cpu().numpy(), p)
                num_bboxes = len(res[1].bbox)

                # Showboundin boxes
                for bbox_id in range(num_bboxes):
                    # if res[1].get_field("labels")[bbox_id] != 1 and 'of' in p:
                    #     continue
                    startX = int(res[1].bbox[bbox_id][0].item())
                    startY = int(res[1].bbox[bbox_id][1].item())
                    endX = int(res[1].bbox[bbox_id][2].item())
                    endY = int(res[1].bbox[bbox_id][3].item())
                    cv2.rectangle(img_to_show, (startX, startY), (endX, endY), self.colors_dict[p_id], 2)
                    score = res[1].get_field('scores')[bbox_id].item()
                    cv2.putText(img_to_show, p + " " + str(np.round(score, 2)), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors_dict[p_id], 2)
                
                self.bbox_predictions[i][p] = (res, self.glip_model.model.entities)
            
            ax.imshow(img_to_show)
        plt.show()
    
    def predict_exact_masks(self):
        print("Feeding the bouning boxes to SAM...")
        
        self.masks_predictions = []
        fig, axs = plt.subplots(3, 4, figsize=(30,22))
        # fig, axs = plt.subplots(1, 2, figsize=(80,22))

        for i in range(len(self.rendered_images)):
            self.masks_predictions.append({})
            
            img = (self.rendered_images[i].permute([1, 2, 0]) * 255.0).to(torch.uint8).cpu().numpy().copy()
            self.sam_model.set_img(img)
            ax = axs[i // 4, i % 4]
            # ax = axs[i]
            ax.imshow(img)
            
            for p_id, p in enumerate(self.prompts):
                print('SAM - View:', i, 'Prompt:', p_id)
    
                glip_res = self.bbox_predictions[i][p][0]
                num_bboxes = len(glip_res[1].bbox)

                masks = []
                # Predict and Show masks
                for bbox_id in range(num_bboxes):
                    startX = int(glip_res[1].bbox[bbox_id][0].item())
                    startY = int(glip_res[1].bbox[bbox_id][1].item())
                    endX = int(glip_res[1].bbox[bbox_id][2].item())
                    endY = int(glip_res[1].bbox[bbox_id][3].item())
                    cv2.rectangle(img, (startX, startY), (endX, endY), self.colors_dict[p_id], 2)
                    box = np.array([startX, startY, endX, endY])
                    mask, score = self.sam_model.predict(
                        bbox=box,
                        multimask_output=False,
                    )
                    # score *= glip_res[1].get_field('scores')[bbox_id].item()
                    if ('mattress' in p and glip_res[1].get_field('labels')[bbox_id] == 1):
                        self.show_mask(mask, ax, np.array(self.colors_dict[p_id]) / 255.0)
                    masks.append((mask, score))
                self.masks_predictions[i][p] = masks
        plt.show()

    def show_mask(self, mask, ax, input_color = None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            if input_color is None:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            color = np.array([input_color[0], input_color[1], input_color[2], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, color):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        # ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))

    def __call__(self):
        assert self.glip_model is not None
        assert self.rendered_images is not None

        return self.predict_face_cls()

class SATRSAM(GLIPSAMMeshSegmenter):
    def __init__(self, cfg):
        super().__init__(cfg)

        (
            self.face_adjacency,
            self.faces_distance_factors,
            self.center_vert_id,
            self.face_visibilty_ratios,
        ) = (None, None, None, None)

    def set_mesh(self, mesh):
        super().set_mesh(mesh)

        print(f"Getting faces neighborhood")
        self.get_faces_neighborhood()
        print(f"Computing vertices pairwise distances")
        self.compute_vertices_pairwise_dist()

    def get_faces_neighborhood(self):
        n = self.cfg.satr.face_smoothing_n_ring

        m = trimesh.Trimesh(
            vertices=self.mesh.vertices.cpu().numpy(),
            faces=self.mesh.faces.cpu().numpy(),
        )

        faces_adjacency = defaultdict(set)

        for el in m.face_adjacency:
            u, v = el
            faces_adjacency[u].add(v)
            faces_adjacency[v].add(u)

        c = []
        faces_adjacency_res = deepcopy(faces_adjacency)

        for k in faces_adjacency:
            for i in range(n - 1):
                start = deepcopy(faces_adjacency_res[k])
                end = set(deepcopy(faces_adjacency_res[k]))
                for f in start:
                    end.update(faces_adjacency[f])
                faces_adjacency_res[k] = end
            c.append(len(faces_adjacency_res[k]))

        self.face_adjacency = faces_adjacency_res

    def compute_faces_to_capital_face_distances(self, faces_dict):
        visible_faces_ids = torch.tensor(sorted(list(faces_dict.keys()))).int().cuda()
        visible_faces = torch.index_select(self.mesh.faces, 0, visible_faces_ids)
        visible_vertices_ids = visible_faces.flatten().unique()

        # Get the mean vertex position of all the faces visible
        v1 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 0])
        v2 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 1])
        v3 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 2])

        visible_face_areas = torch.index_select(
            self.mesh.face_areas, 0, visible_faces_ids
        ).view(len(visible_faces_ids), 1)

        faces_center = ((v1 + v2 + v3) / 3.0) * visible_face_areas

        capital_face_center = faces_center.sum(dim=0) / visible_face_areas.sum(dim=0)

        # Find the nearest vertex to the capital_face_center
        distances = cdist(
            self.mesh.vertices[visible_vertices_ids].cpu().numpy(),
            capital_face_center.cpu().view(1, 3).numpy(),
        )

        nearest_vert_ind = np.argmin(distances[:, 0])
        nearest_vert_id = visible_vertices_ids[nearest_vert_ind]

        # Now compute the distance from this vertex (representing the capital face on the mesh) 
        # to each face found in faces_dict
        faces_distance = {}
        distances = []

        for k, _ in faces_dict.items():
            face_vert = self.mesh.faces[k][0].cpu().numpy()
            distances.append(self.vertices_distances[face_vert, nearest_vert_id])
            faces_distance[k] = self.vertices_distances[face_vert, nearest_vert_id]

        distances = np.array(distances)
        mean = np.mean(distances)
        std = np.std(distances)
        g = scipy.stats.norm(mean, std)

        faces_distance_factor = {}
        for k, v in faces_dict.items():
            faces_distance_factor[k] = g.pdf(faces_distance[k])

        return faces_distance_factor, nearest_vert_id

    def compute_vertices_pairwise_dist(self):
        n_vertices = len(self.mesh.vertices)

        solver = pp3d.MeshHeatMethodDistanceSolver(
            self.mesh.vertices.cpu().numpy(), self.mesh.faces.cpu().numpy()
        )

        self.vertices_distances = np.zeros((n_vertices, n_vertices))
        for i in range(n_vertices):
            x = solver.compute_distance(i)
            self.vertices_distances[:, i] = x

    def preprocessing_step_reweighting_factors(self, included_face_ids):
        if self.cfg.satr.gaussian_reweighting:
            (
                self.faces_distance_factors,
                self.center_vert_id,
            ) = self.compute_faces_to_capital_face_distances(included_face_ids)

        if self.cfg.satr.face_smoothing:
            self.face_visibilty_ratios = self.compute_face_visibilty_ratio(
                included_face_ids
            )

    def compute_reweighting_factors(self, included_face_ids):
        ret = {"faces_distance_factor": {}, "face_visibilty_ratio": {}}

        for k, _ in included_face_ids.items():
            ret["faces_distance_factor"][k] = (
                self.faces_distance_factors[k]
                if self.cfg.satr.gaussian_reweighting
                else 1.0
            )
            ret["face_visibilty_ratio"][k] = (
                self.face_visibilty_ratios[k] if self.cfg.satr.face_smoothing else 1.0
            )

        return ret

    def compute_face_visibilty_ratio(self, relevant_face_ids):
        face_neighborhood = self.face_adjacency

        relevant_face_scores = {}

        for k, v in relevant_face_ids.items():
            score = 0
            for f in face_neighborhood[k]:
                score += f in relevant_face_ids
            score /= len(face_neighborhood[k]) + 0.00001
            relevant_face_scores[k] = score

        return relevant_face_scores

    def process_box_predictions(self, prompt, preds, rendering_face_ids):
        # Initialize a score vector for each face in the mesh for the given region prompt (e.g. "the leg of a person").
        face_view_prompt_score = np.zeros((len(self.mesh.faces)))
        face_view_freq = np.zeros((len(self.mesh.faces)))

        pred_bboxes = preds[0][1]
        pred_bboxes_cor = preds[0][1].bbox
        n_boxes = len(
            pred_bboxes
        )  # The number of predcited bounding boxes for the given prompt (e.g., the leg of a person).

        face_counter = rendering_face_ids.flatten().cpu().int().numpy()
        face_counter = face_counter[face_counter != -1]
        face_counter = Counter(list(face_counter))

        for i in range(n_boxes):
            if pred_bboxes.get_field("labels")[i].item() != 1 and "of" in prompt:
                continue

            confidence_score = pred_bboxes.get_field("scores")[i].item()

            # Get the included face ids inside this bounding box
            included_face_ids, not_included_face_ids = self.get_included_face_ids(
                rendering_face_ids, pred_bboxes_cor[i], face_counter
            )


            # Compute the reweighting factors
            self.preprocessing_step_reweighting_factors(
                included_face_ids,
            )
            reweighting_factors = self.compute_reweighting_factors(included_face_ids)

            for k, v in included_face_ids.items():
                final_factor = 1.0

                for f in list(reweighting_factors.values()):
                    final_factor *= f[k]

                face_view_prompt_score[k] += v * confidence_score * final_factor

        return face_view_prompt_score, face_view_freq
    
    def process_masks_predictions(self, prompt, pred_masks, pred_bboxes, rendering_face_ids):
        
        #pred_masks is a list of tuples. Each tuple correspnds to one mask. 
        #First is a res*res array representing the mask, second is float representing the score

        # Initialize a score vector for each face in the mesh for the given region prompt (e.g. "the leg of a person").
        face_view_prompt_score = np.zeros((len(self.mesh.faces)))
        face_view_freq = np.zeros((len(self.mesh.faces)))


        n_masks = len(
            pred_masks
        )  # The number of predcited bounding boxes for the given prompt (e.g., the leg of a person).

        face_counter = rendering_face_ids.flatten().cpu().int().numpy()
        face_counter = face_counter[face_counter != -1]
        face_counter = Counter(list(face_counter))

        # Traverse masks 
        for i in range(n_masks):
            if pred_bboxes.get_field("labels")[i].item() != 1 and "of" in prompt:
                continue
            confidence_score = pred_masks[i][1] * pred_bboxes.get_field('scores')[i].item()

            # Get the included face ids inside this bounding box
            included_face_ids, not_included_face_ids = self.get_included_face_ids_from_mask(
                rendering_face_ids, pred_masks[i][0], pred_bboxes.bbox[i], face_counter
            )

            if(len(included_face_ids) == 0):
                continue

            # Compute the reweighting factors
            self.preprocessing_step_reweighting_factors(
                included_face_ids,
            )
            reweighting_factors = self.compute_reweighting_factors(included_face_ids)

            for k, v in included_face_ids.items():
                final_factor = 1.0

                for f in list(reweighting_factors.values()):
                    final_factor *= f[k]

                face_view_prompt_score[k] += v * confidence_score * final_factor

        return face_view_prompt_score, face_view_freq

class GLIPMeshSegmenter(BaseDetMeshSegmentor):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.glip_model = GLIPModel()

    def predict_face_cls(self):
        # Get the bounding boxes predictions for the given prompts
        self.predict_bboxes()
        assert self.bbox_predictions is not None
        print(f"Finished GLIP")

        face_cls = np.zeros((len(self.mesh.faces), len(self.prompts)))
        face_freq = np.zeros((len(self.mesh.faces), len(self.prompts)))

        # Looping over the views
        for i, view in tqdm(enumerate(self.rendered_images)):
            for j, prompt in enumerate(self.prompts):
                print(f'Processing view: {i}, Prompt: {j}')
                (
                    face_view_prompt_score,
                    face_view_prompt_freq,
                ) = self.process_box_predictions(
                    prompt,
                    self.bbox_predictions[i][prompt],
                    self.rendered_images_face_ids[i].squeeze(0),
                )

                face_cls[:, j] += face_view_prompt_score
                face_freq[:, j] += face_view_prompt_freq

        return face_cls, face_freq

    def predict_bboxes(self):
        # Generate GLIP predictions for every rendered image
        print("Feeding the views to GLIP...")
        self.bbox_predictions = []
        num_views = len(self.rendered_images)

        # fig, axs = plt.subplots(2,5,figsize=(80,22))
        fig, axs = plt.subplots(3,4,figsize=(80,22))

        colors_dict = {
            0: [255, 0, 0],   # Red
            1: [0, 255, 0],   # Green
            2: [0, 0, 255],   # Blue
            3: [255, 255, 0],   # Yellow
            4: [255, 0, 255],   # Magenta
            5: [0, 255, 255],   # Cyan
            6: [128, 0, 0], # Dark Red
            7: [0, 128, 0], # Dark Green
            8: [0, 0, 128], # Dark Blue
            9: [128, 128, 128] # Gray
        }

        for i in range(len(self.rendered_images)):
            self.bbox_predictions.append({})

            img = self.rendered_images[i].permute([1, 2, 0]) * 256.0
            img = img.to(torch.uint8)
            img_to_show = img.cpu().numpy().copy()
            ax = axs[i // 4, i % 4]
            for p_idx, p in enumerate(self.prompts):
                print('View number:', i, 'Prompt:', p_idx, end=' ')
                
                res = self.glip_model.predict(img.cpu().numpy(), p)
                
                num_bboxes = len(res[1].bbox)

                for bbox_idx in range(num_bboxes):
                    startX = int(res[1].bbox[bbox_idx][0].item())
                    startY = int(res[1].bbox[bbox_idx][1].item())
                    endX = int(res[1].bbox[bbox_idx][2].item())
                    endY = int(res[1].bbox[bbox_idx][3].item())
                    cv2.rectangle(img_to_show, (startX, startY), (endX, endY), colors_dict[p_idx], 2)
                    score = res[1].get_field('scores')[bbox_idx].item()
                    cv2.putText(img_to_show, p + " " + str(score), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors_dict[p_idx], 2)
                
                self.bbox_predictions[i][p] = (res, self.glip_model.model.entities)
            
            ax.imshow(img_to_show)
        plt.show()

    def __call__(self):
        assert self.glip_model is not None
        assert self.rendered_images is not None

        return self.predict_face_cls()

class SATR(GLIPMeshSegmenter):
    def __init__(self, cfg):
        super().__init__(cfg)

        (
            self.face_adjacency,
            self.faces_distance_factors,
            self.center_vert_id,
            self.face_visibilty_ratios,
        ) = (None, None, None, None)

    def set_mesh(self, mesh):
        super().set_mesh(mesh)

        print(f"Getting faces neighborhood")
        self.get_faces_neighborhood()
        print(f"Computing vertices pairwise distances")
        self.compute_vertices_pairwise_dist()

    def get_faces_neighborhood(self):
        n = self.cfg.satr.face_smoothing_n_ring

        m = trimesh.Trimesh(
            vertices=self.mesh.vertices.cpu().numpy(),
            faces=self.mesh.faces.cpu().numpy(),
        )

        faces_adjacency = defaultdict(set)

        for el in m.face_adjacency:
            u, v = el
            faces_adjacency[u].add(v)
            faces_adjacency[v].add(u)

        c = []
        faces_adjacency_res = deepcopy(faces_adjacency)

        for k in faces_adjacency:
            for i in range(n - 1):
                start = deepcopy(faces_adjacency_res[k])
                end = set(deepcopy(faces_adjacency_res[k]))
                for f in start:
                    end.update(faces_adjacency[f])
                faces_adjacency_res[k] = end
            c.append(len(faces_adjacency_res[k]))

        self.face_adjacency = faces_adjacency_res

    def compute_faces_to_capital_face_distances(self, faces_dict):
        visible_faces_ids = torch.tensor(sorted(list(faces_dict.keys()))).int().cuda()
        visible_faces = torch.index_select(self.mesh.faces, 0, visible_faces_ids)
        visible_vertices_ids = visible_faces.flatten().unique()

        # Get the mean vertex position of all the faces visible
        v1 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 0])
        v2 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 1])
        v3 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 2])

        visible_face_areas = torch.index_select(
            self.mesh.face_areas, 0, visible_faces_ids
        ).view(len(visible_faces_ids), 1)

        faces_center = ((v1 + v2 + v3) / 3.0) * visible_face_areas

        capital_face_center = faces_center.sum(dim=0) / visible_face_areas.sum(dim=0)

        # Find the nearest vertex to the capital_face_center
        distances = cdist(
            self.mesh.vertices[visible_vertices_ids].cpu().numpy(),
            capital_face_center.cpu().view(1, 3).numpy(),
        )

        nearest_vert_ind = np.argmin(distances[:, 0])
        nearest_vert_id = visible_vertices_ids[nearest_vert_ind]

        # Now compute the distance from this vertex (representing the capital face on the mesh) 
        # to each face found in faces_dict
        faces_distance = {}
        distances = []

        for k, _ in faces_dict.items():
            face_vert = self.mesh.faces[k][0].cpu().numpy()
            distances.append(self.vertices_distances[face_vert, nearest_vert_id])
            faces_distance[k] = self.vertices_distances[face_vert, nearest_vert_id]

        distances = np.array(distances)
        mean = np.mean(distances)
        std = np.std(distances)
        g = scipy.stats.norm(mean, std)

        faces_distance_factor = {}
        for k, v in faces_dict.items():
            faces_distance_factor[k] = g.pdf(faces_distance[k])

        return faces_distance_factor, nearest_vert_id

    def compute_vertices_pairwise_dist(self):
        n_vertices = len(self.mesh.vertices)

        solver = pp3d.MeshHeatMethodDistanceSolver(
            self.mesh.vertices.cpu().numpy(), self.mesh.faces.cpu().numpy()
        )

        self.vertices_distances = np.zeros((n_vertices, n_vertices))
        for i in range(n_vertices):
            x = solver.compute_distance(i)
            self.vertices_distances[:, i] = x

    def preprocessing_step_reweighting_factors(self, included_face_ids):
        if self.cfg.satr.gaussian_reweighting:
            (
                self.faces_distance_factors,
                self.center_vert_id,
            ) = self.compute_faces_to_capital_face_distances(included_face_ids)

        if self.cfg.satr.face_smoothing:
            self.face_visibilty_ratios = self.compute_face_visibilty_ratio(
                included_face_ids
            )

    def compute_reweighting_factors(self, included_face_ids):
        ret = {"faces_distance_factor": {}, "face_visibilty_ratio": {}}

        for k, _ in included_face_ids.items():
            ret["faces_distance_factor"][k] = (
                self.faces_distance_factors[k]
                if self.cfg.satr.gaussian_reweighting
                else 1.0
            )
            ret["face_visibilty_ratio"][k] = (
                self.face_visibilty_ratios[k] if self.cfg.satr.face_smoothing else 1.0
            )

        return ret

    def compute_face_visibilty_ratio(self, relevant_face_ids):
        face_neighborhood = self.face_adjacency

        relevant_face_scores = {}

        for k, v in relevant_face_ids.items():
            score = 0
            for f in face_neighborhood[k]:
                score += f in relevant_face_ids
            score /= len(face_neighborhood[k]) + 0.00001
            relevant_face_scores[k] = score

        return relevant_face_scores

    def process_box_predictions(self, prompt, preds, rendering_face_ids):
        # Initialize a score vector for each face in the mesh for the given region prompt (e.g. "the leg of a person").
        face_view_prompt_score = np.zeros((len(self.mesh.faces)))
        face_view_freq = np.zeros((len(self.mesh.faces)))

        pred_bboxes = preds[0][1]
        pred_bboxes_cor = preds[0][1].bbox
        n_boxes = len(
            pred_bboxes
        )  # The number of predcited bounding boxes for the given prompt (e.g., the leg of a person).

        face_counter = rendering_face_ids.flatten().cpu().int().numpy()
        face_counter = face_counter[face_counter != -1]
        face_counter = Counter(list(face_counter))

        for i in range(n_boxes):
            if pred_bboxes.get_field("labels")[i].item() != 1 and "of" in prompt:
                continue

            confidence_score = pred_bboxes.get_field("scores")[i].item()

            # Get the included face ids inside this bounding box
            included_face_ids, not_included_face_ids = self.get_included_face_ids(
                rendering_face_ids, pred_bboxes_cor[i], face_counter
            )

            # Compute the reweighting factors
            self.preprocessing_step_reweighting_factors(
                included_face_ids,
            )
            reweighting_factors = self.compute_reweighting_factors(included_face_ids)

            for k, v in included_face_ids.items():
                final_factor = 1.0

                for f in list(reweighting_factors.values()):
                    final_factor *= f[k]

                face_view_prompt_score[k] += v * confidence_score * final_factor

        return face_view_prompt_score, face_view_freq