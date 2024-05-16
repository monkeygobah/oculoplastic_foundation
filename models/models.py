import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from .config import SAM_CHECKPOINT_PATH, SAM_ENCODER_VERSION, DEVICE  
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt


class SAM:
    def __init__(self):
        self.predictor = self._initialize_predictor()

    def _initialize_predictor(self):
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
        return SamPredictor(sam)

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)

    def segment_no_jitter(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        result_array = np.array(result_masks) 
        return {'right_iris': np.transpose(np.nonzero(result_array[0])),
                'left_iris': np.transpose(np.nonzero(result_array[1])),
                'right_pupil': np.transpose(np.nonzero(result_array[2])),
                'left_pupil': np.transpose(np.nonzero(result_array[3])),
                'right_sclera': np.transpose(np.nonzero(result_array[4])),
                'left_sclera': np.transpose(np.nonzero(result_array[5])),
                'right_brow': np.transpose(np.nonzero(result_array[6])),
                'left_brow': np.transpose(np.nonzero(result_array[7]))
                }, {'right_iris': result_array[0],
                'left_iris':  result_array[1],
                'right_pupil': result_array[2],
                'left_pupil': result_array[3],
                'right_sclera': result_array[4],
                'left_sclera': result_array[5],
                'right_brow': result_array[6],
                'left_brow': result_array[7]}


    def segment_no_jitter_l_sr(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        result_array = np.array(result_masks) 
        return {
                'left_iris': np.transpose(np.nonzero(result_array[0])),
                'left_pupil': np.transpose(np.nonzero(result_array[1])),
                'left_sclera': np.transpose(np.nonzero(result_array[2]))
                }, {
                'left_iris':  result_array[0],
                'left_pupil': result_array[1],
                'left_sclera': result_array[2]
                   }
        
    def segment_no_jitter_r_sr(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        result_array = np.array(result_masks) 
        return {
                'right_iris': np.transpose(np.nonzero(result_array[0])),
                'right_pupil': np.transpose(np.nonzero(result_array[1])),
                'right_sclera': np.transpose(np.nonzero(result_array[2]))
                }, {
                'right_iris':  result_array[0],
                'right_pupil': result_array[1],
                'right_sclera': result_array[2]
                   }

    def segment_no_jitter_brow(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        result_array = np.array(result_masks) 
        return {
                'right_brow': np.transpose(np.nonzero(result_array[0])),
                'left_brow': np.transpose(np.nonzero(result_array[1])),
                }, {
                'right_brow':  result_array[0],
                'left_brow': result_array[1]
                }
        
    def segment_jitter(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        result_array = np.array(result_masks) 
        return {'orig': np.transpose(np.nonzero(result_array[0])),
                'up': np.transpose(np.nonzero(result_array[1])),
                'down': np.transpose(np.nonzero(result_array[2])),
                'left': np.transpose(np.nonzero(result_array[3])),
                'right': np.transpose(np.nonzero(result_array[4])),
                'top_left': np.transpose(np.nonzero(result_array[5])),
                'top_right': np.transpose(np.nonzero(result_array[6])),
                'bottom_left': np.transpose(np.nonzero(result_array[7])),
                'bottom_right': np.transpose(np.nonzero(result_array[8]))
               }, {'orig': result_array[0],
                'up':  result_array[1],
                'down': result_array[2],
                'left': result_array[3],
                'right': result_array[4],
                'top_left': result_array[5],
                'top_right': result_array[6],
                'bottom_left': result_array[7],
                'bottom_right': result_array[8]
               }
        
    def get_embeddings(self, image):
        self.set_image(image)
        image_features = self.predictor.get_image_embedding()
        return image_features
        
    # helper functions to show mask and box for SAM display 
    @staticmethod
    def show_mask(mask, plt, color, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image)
        
    @staticmethod
    #display the bounding box on the plot
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

 
class MP:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFacemesh = mp.solutions.face_mesh
    
    def map(self, img, name):
        # img = cv2.imread(image_path)
        self.faceMesh = self.mpFacemesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
        drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        results = self.faceMesh.process(img)
        id_list = []
        if results.multi_face_landmarks:
            for landmark in results.multi_face_landmarks:
                # self.mpDraw.draw_landmarks(img, landmark, self.mpFacemesh.FACEMESH_TESSELATION, drawSpec,drawSpec)
                for id,lm in enumerate(landmark.landmark):
                    [ih, iw, ic] = img.shape
                    px,py,pz =int(lm.x*iw), int(lm.y*ih), (lm.z*iw)
                    append = [px,py]
                    id_list.append(append)   

        # cv2.imwrite(f"../outputs/mp_landmarks/{name}_mp_landmarks.jpg", img)


        isClosed=True
        return id_list

    def close(self):
        self.faceMesh.close()
