import numpy as np
from  imageTools import CropImages, IncreaseContrast, GammaCorrection, AdaptiveHistogramEqualization
import cv2
from models.config import PLOT_CONFIG, ANALYZE_CONFIG



# TODO: getrid of magic numbers and define the fm coordinates with human readable words

class BoxMaker:
    def __init__(self, image, fm, file):
        self.image = image
        self.fm = fm
        self.file = file
        self.cropped_image, self.crop_coords = CropImages.crop_image(image, fm, file)

    def _adjust_boxes(self, bounding_box):
        x1_crop, y1_crop, x2_crop, y2_crop =  self.crop_coords
        # adjusted_boxes = []
        # for box in bounding_boxes:
        x1_box, y1_box, x2_box, y2_box = bounding_box
        x1_box_adj = x1_box - x1_crop
        y1_box_adj = y1_box - y1_crop
        x2_box_adj = x2_box - x1_crop
        y2_box_adj = y2_box - y1_crop
        adjusted_box = np.array([x1_box_adj, y1_box_adj, x2_box_adj, y2_box_adj])
        
        # adjusted_box = np.array(adjusted_box)
        # adjusted_boxes.append(adjusted_box)
        
        return adjusted_box
    
    def get_all_boxes(self):
            r_iris_box, l_iris_box = self.iris_box()
            r_pupil_box, l_pupil_box = self.pupil_box()
            r_sclera_box, l_sclera_box = self.sclera_box()
            r_brow_box, l_brow_box = self.brow_box()
            midline = self.midline_box()
            crop_image = self.get_cropped_image()
        
            # if ANALYZE_CONFIG['image_alter']['method'] == 'contrast':
            #     contrast_image = IncreaseContrast.convert_colors(crop_image)
            #     return [r_iris_box, l_iris_box, r_pupil_box, l_pupil_box, r_sclera_box, l_sclera_box, r_brow_box, l_brow_box, midline], crop_image, contrast_image  
            
            # After cropping and before passing the image to SAM
            if ANALYZE_CONFIG['image_alter']['method'] == 'gamma_correction':
                gamma = ANALYZE_CONFIG['image_alter']['params']['gamma']
                gamma_image = GammaCorrection.apply(crop_image, gamma)
                return [r_iris_box, l_iris_box, r_pupil_box, l_pupil_box, r_sclera_box, l_sclera_box, r_brow_box, l_brow_box, midline], crop_image, gamma_image, self.crop_coords    

            elif ANALYZE_CONFIG['image_alter']['method'] == 'adaptive_histogram':
                clipLimit = ANALYZE_CONFIG['image_alter']['params']['clipLimit']
                tileGridSize = ANALYZE_CONFIG['image_alter']['params']['tileGridSize']
                adapt_hist_image = AdaptiveHistogramEqualization.apply(crop_image, clipLimit, tileGridSize)
                return [r_iris_box, l_iris_box, r_pupil_box, l_pupil_box, r_sclera_box, l_sclera_box, r_brow_box, l_brow_box, midline], crop_image, adapt_hist_image, self.crop_coords     
                
            elif ANALYZE_CONFIG['image_alter']['method'] == 'contrast':
                contrast_image = IncreaseContrast.apply(crop_image)
                return [r_iris_box, l_iris_box, r_pupil_box, l_pupil_box, r_sclera_box, l_sclera_box, r_brow_box, l_brow_box, midline], crop_image, contrast_image, self.crop_coords      
                
            else:
                return [r_iris_box, l_iris_box, r_pupil_box, l_pupil_box, r_sclera_box, l_sclera_box, r_brow_box, l_brow_box, midline], crop_image, self.crop_coords    
        
    def iris_box(self):
        iris_pad = 3
        r_iris_bb_top_right = [self.fm[471][0] - iris_pad, self.fm[470][1] - iris_pad]
        r_iris_bb_bottom_left = [self.fm[469][0] + iris_pad, self.fm[472][1] + iris_pad]
        
        l_iris_bb_top_right = [self.fm[476][0] - iris_pad, self.fm[475][1] - iris_pad]
        l_iris_bb_bottom_left = [self.fm[474][0] + iris_pad, self.fm[477][1] + iris_pad]
        
        r_iris_bb = [r_iris_bb_top_right[0], r_iris_bb_top_right[1], r_iris_bb_bottom_left[0], r_iris_bb_bottom_left[1]]
        l_iris_bb = [l_iris_bb_top_right[0], l_iris_bb_top_right[1], l_iris_bb_bottom_left[0], l_iris_bb_bottom_left[1]]

        return self._adjust_boxes(r_iris_bb),  self._adjust_boxes(l_iris_bb)   
   
    def pupil_box(self):
         # 469 -> left, 470 -> top, 471 -> right,  472-> bottom,
        r_pupil_top_right = [(self.fm[468][0] + self.fm[471][0])/2, (self.fm[468][1] + self.fm[470][1])/2]
        r_pupil_bottom_left = [(self.fm[468][0] + self.fm[469][0])/2, (self.fm[468][1] + self.fm[472][1])/2]

        # 474 -> left, 475 -> top, 476 -> right,  477-> bottom,
        l_pupil_top_right = [(self.fm[473][0] + self.fm[476][0])/2, (self.fm[473][1] + self.fm[475][1])/2]
        l_pupil_bottom_left = [(self.fm[473][0] + self.fm[474][0])/2, (self.fm[473][1] + self.fm[477][1])/2]
        
        r_pupil_bb = [r_pupil_top_right[0], r_pupil_top_right[1], r_pupil_bottom_left[0], r_pupil_bottom_left[1]]
        l_pupil_bb = [l_pupil_top_right[0], l_pupil_top_right[1], l_pupil_bottom_left[0], l_pupil_bottom_left[1]]
        
        return self._adjust_boxes(r_pupil_bb), self._adjust_boxes(l_pupil_bb)       
 
    def sclera_box(self):
        sclera_pad_y = 5
        sclera_pad_x = 0
        r_sclera_bb_top_right = [self.fm[130][0] - sclera_pad_x , self.fm[159][1]- sclera_pad_y]
        r_sclera_bb_bottom_left = [self.fm[133][0] + sclera_pad_x, self.fm[145][1]+ sclera_pad_y]
        
        l_sclera_bb_top_right = [self.fm[362][0]- sclera_pad_x, self.fm[386][1]- sclera_pad_y]
        l_sclera_bb_bottom_left = [self.fm[359][0]+ sclera_pad_x, self.fm[145][1]+ sclera_pad_y]
    
        r_sclera_bb = [r_sclera_bb_top_right[0], r_sclera_bb_top_right[1], r_sclera_bb_bottom_left[0], r_sclera_bb_bottom_left[1]]
        l_sclera_bb = [l_sclera_bb_top_right[0], l_sclera_bb_top_right[1], l_sclera_bb_bottom_left[0], l_sclera_bb_bottom_left[1]]
        
        return self._adjust_boxes(r_sclera_bb), self._adjust_boxes(l_sclera_bb)    
    
    def brow_box(self):
        eyebrow_pad_y_superior = 20
        eyebrow_pad_y_inferior = 0
        eyebrow_pad_x =0 
        r_eyebrow_bb_top_right = [self.fm[70][0] - eyebrow_pad_x , self.fm[105][1]- eyebrow_pad_y_superior]
        r_eyebrow_bb_bottom_left = [self.fm[55][0] + eyebrow_pad_x, self.fm[65][1]+ eyebrow_pad_y_inferior]
        
        l_eyebrow_bb_top_right = [self.fm[285][0] - eyebrow_pad_x , self.fm[296][1]- eyebrow_pad_y_superior]
        l_eyebrow_bb_bottom_left = [self.fm[276][0] + eyebrow_pad_x , self.fm[283][1]+eyebrow_pad_y_inferior]
        
        r_brow_bb = [r_eyebrow_bb_top_right[0], r_eyebrow_bb_top_right[1], r_eyebrow_bb_bottom_left[0], r_eyebrow_bb_bottom_left[1]]
        l_brow_bb = [l_eyebrow_bb_top_right[0], l_eyebrow_bb_top_right[1], l_eyebrow_bb_bottom_left[0], l_eyebrow_bb_bottom_left[1]]
        
        return self._adjust_boxes(r_brow_bb), self._adjust_boxes(l_brow_bb)     
    
    
    def midline_box(self):
        nasion = self.fm[168]
        mid_hairline= self.fm[10]
        midline = [nasion[0], nasion[1], mid_hairline[0], mid_hairline[1]]
        return self._adjust_boxes(midline)    

    def get_cropped_image(self):

        return self.cropped_image

