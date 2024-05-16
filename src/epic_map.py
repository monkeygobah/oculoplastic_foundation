import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import utils_epic
import pandas as pd

def epicmap(image_name, file, crop_coords, cropped_image): 
    fm,  img, ll0_data,  lu0_data,  rl0_data,  ru0_data,  left_iris, right_iris = utils_epic.map_tesselations_epic(image_name, crop_coords)

    
    # extract all the important stuff from map tesselations
    left_upper_lid_x = lu0_data[1]
    left_upper_lid_y= lu0_data[0]
    left_lower_lid_x= ll0_data[1]
    left_lower_lid_y= ll0_data[0]
    right_upper_lid_x= ru0_data[1]
    right_upper_lid_y= ru0_data[0]
    right_lower_lid_x= rl0_data[1]
    right_lower_lid_y= rl0_data[0]
    initial_tl= lu0_data[3]
    initial_bl= ll0_data[3]
    initial_tr= ru0_data[3]
    initial_br= rl0_data[3]
    coef_tl= lu0_data[4]
    coef_bl= ll0_data[4]
    coef_tr= ru0_data[4]
    coef_br= rl0_data[4]
    
     ## get min and max x value of upper left lid to generate lots of points
    _, left_upper_lid_x, left_upper_lid_y, l_upper_lid, l_u = utils_epic.calculateLids(left_upper_lid_x, initial_tl, coef_tl)
    _, right_upper_lid_x, right_upper_lid_y, r_upper_lid, r_u = utils_epic.calculateLids(right_upper_lid_x, initial_tr, coef_tr)   
    _, left_lower_lid_x, left_lower_lid_y, l_lower_lid, l_l = utils_epic.calculateLids(left_lower_lid_x, initial_bl, coef_bl)            
    _, right_lower_lid_x, right_lower_lid_y, r_lower_lid, r_l = utils_epic.calculateLids(right_lower_lid_x, initial_br, coef_br)
    
    left_lid_tracing = np.concatenate((np.array(l_upper_lid), np.array(l_lower_lid)))
    right_lid_tracing = np.concatenate((np.array(r_upper_lid), np.array(r_lower_lid)))

    
    # define canthi from mediapipe   
    l_medial_canthus = np.array(fm[362])
    l_lateral_canthus = np.array(fm[359])
    r_medial_canthus = np.array(fm[133])
    r_lateral_canthus = np.array(fm[130])
    
    left_iris_centroid, left_iris_diameter, left_iris_superior, left_iris_inferior = utils_epic.get_iris_info(np.array(left_iris), cropped_image)
    right_iris_centroid, right_iris_diameter, right_iris_superior, right_iris_inferior = utils_epic.get_iris_info(np.array(right_iris), cropped_image)
    # cf = 11.7 / ((right_iris_diameter+left_iris_diameter)/2)
    

    left_lid_superior, left_lid_inferior = utils_epic.get_vertical_extreme_points(left_lid_tracing, left_iris_superior, left_iris_inferior,cropped_image)
    right_lid_superior, right_lid_inferior = utils_epic.get_vertical_extreme_points(right_lid_tracing, right_iris_superior, right_iris_inferior,cropped_image)
    
    
    left_upper_brow_x_np = np.array([fm[383][0], fm[300][0], fm[293][0], fm[334][0], fm[296][0], fm[336][0], fm[285][0], fm[417][0]])
    left_upper_brow_y_np = np.array([fm[383][1], fm[300][1], fm[293][1], fm[334][1], fm[296][1], fm[336][1], fm[285][1], fm[417][1]])
    right_upper_brow_x_np = np.array([fm[156][0], fm[70][0], fm[63][0], fm[105][0], fm[66][0], fm[107][0], fm[55][0], fm[193][0]])
    right_upper_brow_y_np = np.array([fm[156][1], fm[70][1], fm[63][1], fm[105][1], fm[66][1], fm[107][1], fm[55][1], fm[193][1]])
    
    left_lower_brow_x = np.array([fm[285][0], fm[295][0], fm[282][0], fm[283][0], fm[276][0], fm[383][0]])
    left_lower_brow_y = np.array([fm[285][1], fm[295][1], fm[282][1], fm[283][1], fm[276][1], fm[383][1]])
    right_lower_brow_x = np.array([fm[55][0], fm[65][0], fm[52][0], fm[53][0], fm[46][0], fm[124][0]])
    right_lower_brow_y = np.array([fm[55][1], fm[65][1], fm[52][1], fm[53][1], fm[46][1], fm[124][1]])
    

    right_brow,  r2,  right_lower_brow_x, right_lower_brow_y =  utils_epic.calculateBrows(right_lower_brow_x, right_lower_brow_y)
    left_brow,  l2,  left_lower_brow_x, left_lower_brow_y = utils_epic.calculateBrows( left_lower_brow_x, left_lower_brow_y)     


    right_brow_sup,  r2_sup,  right_lower_brow_x_sup, right_lower_brow_y_sup =  utils_epic.calculateBrows(right_upper_brow_x_np, right_upper_brow_y_np)
    left_brow_sup,  l2_sup,  left_lower_brow_x_sup, left_lower_brow_y_sup = utils_epic.calculateBrows( left_upper_brow_x_np, left_upper_brow_y_np)   

    #find points on eyebrows
    l_lat_eyebrow, l_center_eyebrow, l_medial_eyebrow, r_lat_eyebrow, r_center_eyebrow, r_medial_eyebrow = \
        utils_epic.get_eyebrow_points(np.array(left_brow), np.array(right_brow), l_lateral_canthus, r_lateral_canthus, l_medial_canthus, r_medial_canthus, right_iris_centroid, left_iris_centroid)  


    #find points on upper eyebrows
    l_lat_eyebrow_sup, l_center_eyebrow_sup, l_medial_eyebrow_sup, r_lat_eyebrow_sup, r_center_eyebrow_sup, r_medial_eyebrow_sup = \
        utils_epic.get_eyebrow_points(np.array(left_brow_sup), np.array(right_brow_sup), l_lateral_canthus, r_lateral_canthus, l_medial_canthus, r_medial_canthus, right_iris_centroid, left_iris_centroid)  
    
    left_horiz_pf =np.linalg.norm([l_medial_canthus[0], l_lateral_canthus[1]]- l_lateral_canthus) 
    right_horiz_pf =np.linalg.norm([r_medial_canthus[0], r_lateral_canthus[1]]- r_lateral_canthus) 
    
    left_mc_bh = np.linalg.norm(l_medial_canthus-l_medial_eyebrow)
    left_central_bh =np.linalg.norm(left_iris_centroid-l_center_eyebrow) 
    left_lc_bh =np.linalg.norm(l_lateral_canthus-l_lat_eyebrow) 

    sup_left_mc_bh = np.linalg.norm(l_medial_canthus-l_medial_eyebrow_sup)
    sup_left_central_bh = np.linalg.norm(left_iris_centroid-l_center_eyebrow_sup)
    sup_left_lc_bh = np.linalg.norm(l_lateral_canthus-l_lat_eyebrow_sup)
    
    right_mc_bh = np.linalg.norm(r_medial_canthus-r_medial_eyebrow) 
    right_central_bh = np.linalg.norm(right_iris_centroid-r_center_eyebrow) 
    right_lc_bh = np.linalg.norm(r_lateral_canthus-r_lat_eyebrow) 

    sup_right_mc_bh = np.linalg.norm(r_medial_canthus-r_medial_eyebrow_sup) 
    sup_right_central_bh = np.linalg.norm(right_iris_centroid-r_center_eyebrow_sup) 
    sup_right_lc_bh = np.linalg.norm(r_lateral_canthus-r_lat_eyebrow_sup) 

    
    if left_lid_superior[1] > left_iris_superior[1]:
        left_mrd_1 = np.linalg.norm(left_iris_centroid-left_lid_superior) 
        left_SSS = 0
    else:
        left_mrd_1 = np.linalg.norm(left_iris_centroid-left_iris_superior) 
        left_SSS = np.linalg.norm(left_lid_superior-left_iris_superior) 
    
    if left_lid_inferior[1] < left_iris_inferior[1]:
        left_mrd_2 = np.linalg.norm(left_iris_centroid-left_lid_inferior) 
        left_ISS = 0
    else:
        left_mrd_2 = np.linalg.norm(left_iris_centroid-left_iris_inferior) 
        left_ISS = np.linalg.norm(left_lid_inferior-left_iris_inferior) 
    
    left_vert_pf = left_mrd_1 + left_mrd_2
    
    if right_lid_superior[1] > right_iris_superior[1]:
        right_mrd_1 = np.linalg.norm(right_iris_centroid-right_lid_superior) 
        right_SSS = 0
    else:
        right_mrd_1 = np.linalg.norm(right_iris_centroid-right_iris_superior) 
        right_SSS = np.linalg.norm(right_lid_superior-right_iris_superior) 
    
    if right_lid_inferior[1] < right_iris_inferior[1]:
        right_mrd_2 =  np.linalg.norm(right_iris_centroid-right_lid_inferior) 
        right_ISS = 0
    else:
        right_mrd_2 =  np.linalg.norm(right_iris_centroid-right_iris_inferior) 
        right_ISS = np.linalg.norm(right_lid_inferior-right_iris_inferior) 
    
    right_vert_pf = right_mrd_1 + right_mrd_2

    
    icd,ipd,ocd = utils_epic.interCanthalDistance(r_medial_canthus, l_medial_canthus, r_lateral_canthus, l_lateral_canthus, left_iris_centroid, right_iris_centroid)
    
    left_canthal_tilt, right_canthal_tilt = utils_epic.detectCanthalTilt(fm[168],r_lateral_canthus, r_medial_canthus, l_lateral_canthus, l_medial_canthus)
    
    vertical_dystopia, left_vd_point, right_vd_point = utils_epic.detectVerticalDystopia(right_iris_centroid, left_iris_centroid, fm[168])
    
#     df_results = pd.DataFrame({
#     'file': file,
#     'right_iris_diameter': [right_iris_diameter],
#     'left_iris_diameter': [left_iris_diameter],
#     'average_diamter'   : [(right_iris_diameter+left_iris_diameter)/2],
#     'right_mrd_1'       : [right_mrd_1],
#     'right_mrd_2'       : [right_mrd_2],
#     'right_vert_pf'     : [right_vert_pf],
#     'right_horiz_fissure': [right_horiz_pf],
#     'right_SSS'         : [right_SSS],
#     'right_ISS'         : [right_ISS],
#     'right_medial_bh'   : [right_mc_bh],
#     'right_central_bh'  : [right_central_bh],
#     'right_lateral_bh'  : [right_lc_bh],
#     'right_canthal_tilt': [right_canthal_tilt],
#     'left_mrd_1'        : [left_mrd_1],
#     'left_mrd_2'        : [left_mrd_2],
#     'left_vert_pf'      : [left_vert_pf],
#     'left_horiz_fissure': [left_horiz_pf],
#     'left_SSS'          : [left_SSS],
#     'left_ISS'          : [left_ISS],
#     'left_medial_bh'    : [left_mc_bh],
#     'left_central_bh'   : [left_central_bh],
#     'left_lateral_bh'   : [left_lc_bh],
#     'left_canthal_tilt' : [left_canthal_tilt],
#     'icd'               : [icd],
#     'ipd'               : [ipd],
#     'ocd'               : [ocd],
#     'vert_dystopia'     : [vertical_dystopia]
# })

    landmarks = {
         'iris_discrepancy': '', 
         'right_iris_centroid':  right_iris_centroid , 
         'right_iris_diameter':  right_iris_diameter, 
         'right_iris_superior':   right_iris_superior,
         'right_iris_inferior':   right_iris_inferior,
         'left_iris_centroid':    left_iris_centroid, 
         'left_iris_diameter':    left_iris_diameter, 
         'left_iris_superior':    left_iris_superior,
         'left_iris_inferior':    left_iris_inferior,
         'left_sclera_superior':  left_lid_superior, 
         'left_sclera_inferior':  left_lid_inferior, 
         'right_sclera_superior': right_lid_superior, 
         'right_sclera_inferior': right_lid_inferior, 
         'right_medial_canthus': r_medial_canthus , 
         'right_lateral_canthus': r_lateral_canthus, 
         'left_medial_canthus':  l_medial_canthus , 
         'left_lateral_canthus':  l_lateral_canthus, 
         'l_lat_eyebrow':         l_lat_eyebrow,
         'l_center_eyebrow':      l_center_eyebrow,
         'l_medial_eyebrow':      l_medial_eyebrow, 
         'r_lat_eyebrow':         r_lat_eyebrow, 
         'r_center_eyebrow':      r_center_eyebrow,
         'r_medial_eyebrow':      r_medial_eyebrow,
         'sup_l_lat_eyebrow' : l_lat_eyebrow_sup,
         'sup_l_center_eyebrow' : l_center_eyebrow_sup,
         'sup_l_medial_eyebrow' : l_medial_eyebrow_sup,
         'sup_r_lat_eyebrow' : r_lat_eyebrow_sup,
         'sup_r_center_eyebrow' : r_center_eyebrow_sup,
         'sup_r_medial_eyebrow' : r_medial_eyebrow_sup,
         'midline': [fm[168][0], fm[168][1], fm[10][0], fm[10][1]]

     }

    measurements = {
        'left_horiz_fissure': left_horiz_pf, 
        'right_horiz_fissure': right_horiz_pf, 
        'left_medial_bh': left_mc_bh, 
        'left_central_bh': left_central_bh, 
        'left_lateral_bh': left_lc_bh, 
        'right_medial_bh': right_mc_bh, 
        'right_central_bh': right_central_bh, 
        'right_lateral_bh': right_lc_bh, 
        'sup_left_medial_bh' : sup_left_mc_bh,
        'sup_left_central_bh' : sup_left_central_bh,
        'sup_left_lateral_bh' : sup_left_lc_bh,
        'sup_right_medial_bh' : sup_right_mc_bh,
        'sup_right_central_bh' : sup_right_central_bh,
        'sup_right_lateral_bh' : sup_right_lc_bh,
        'left_mrd_1': left_mrd_1 , 
        'left_SSS': left_SSS, 
        'left_mrd_2': left_mrd_2,
        'left_ISS': left_ISS, 
        'right_mrd_1': right_mrd_1, 
        'right_SSS': right_SSS, 
        'right_mrd_2': right_mrd_2, 
        'right_ISS': right_ISS, 
        'right_vert_pf': right_vert_pf, 
        'left_vert_pf': left_vert_pf, 
        'icd': icd, 
        'ipd': ipd, 
        'ocd': ocd, 
        'left_canthal_tilt': left_canthal_tilt, 
        'right_canthal_tilt': right_canthal_tilt, 
        'vert_dystopia': vertical_dystopia, 
        'left_vd_plot_point': left_vd_point, 
        'right_vd_plot_point': right_vd_point
    }




    return measurements, landmarks
    







 # ### PLOTTING
    # # plt.axis('on')
    # tempRotateImage = cropped_image     
    
    # thickness =2
    # # Bottom subplot - image with additional graphics
    # plt.imshow(tempRotateImage)
    # # left brow heights
    # if not np.array_equal(l_medial_eyebrow, np.array([0,0])):
    #     plt.plot([l_medial_canthus[0], l_medial_eyebrow[0]], [l_medial_canthus[1], l_medial_eyebrow[1]], color='black',linewidth=thickness)
    # if not np.array_equal(l_center_eyebrow, np.array([0,0])):
    #     plt.plot([left_iris_centroid[0], l_center_eyebrow[0]], [left_iris_centroid[1], l_center_eyebrow[1]], color='black',linewidth=thickness) 
    # if not np.array_equal(l_lat_eyebrow, np.array([0,0])):
    #     plt.plot([l_lateral_canthus[0], l_lat_eyebrow[0]], [l_lateral_canthus[1], l_lat_eyebrow[1]], color='black',linewidth=thickness)
    
    # # right brow heights
    # if not np.array_equal(r_medial_eyebrow, np.array([0,0])):
    #     plt.plot([r_medial_canthus[0], r_medial_eyebrow[0]], [r_medial_canthus[1], r_medial_eyebrow[1]], color='black',linewidth=thickness)            
    # if not np.array_equal(r_center_eyebrow, np.array([0,0])):
    #     plt.plot([right_iris_centroid[0], r_center_eyebrow[0]], [right_iris_centroid[1], r_center_eyebrow[1]], color='black',linewidth=thickness)
    # if not np.array_equal(r_lat_eyebrow, np.array([0,0])):
    #     plt.plot([r_lateral_canthus[0], r_lat_eyebrow[0]], [r_lateral_canthus[1], r_lat_eyebrow[1]], color='black',linewidth=thickness)
    
    # # left scleral show
    # plt.plot([left_lid_superior[0], left_iris_superior[0]], [left_lid_superior[1], left_iris_superior[1]], color='blue',linewidth=thickness)
    # plt.plot([left_lid_inferior[0], left_iris_inferior[0]], [left_lid_inferior[1], left_iris_inferior[1]], color='orange',linewidth=thickness)
    
    # # # right scleral show
    # plt.plot([right_lid_superior[0], right_iris_superior[0]], [right_lid_superior[1], right_iris_superior[1]], color='blue',linewidth=thickness)
    # plt.plot([right_lid_inferior[0], right_iris_inferior[0]], [right_lid_inferior[1], right_iris_inferior[1]], color='orange',linewidth=thickness)
    
    # print([left_iris_centroid[0], left_iris_superior[0]], [left_iris_centroid[1], left_iris_superior[1]])
    # ## left mrd 
    # # plt.plot([left_iris_centroid[0], left_iris_superior[0]], [left_iris_centroid[1], left_iris_superior[1]], color='lightblue',linewidth=thickness)
    # # plt.plot([left_iris_centroid[0], left_iris_inferior[0]], [left_iris_centroid[1], left_iris_inferior[1]], color='purple',linewidth=thickness)
    
    # #right mrd
    # # plt.plot([right_iris_centroid[0], right_iris_superior[0]], [right_iris_centroid[1], right_iris_superior[1]], color='lightblue',linewidth=thickness)
    # # plt.plot([right_iris_centroid[0], right_iris_inferior[0]], [right_iris_centroid[1], right_iris_inferior[1]], color='purple',linewidth=thickness)
    
    # # vertical dystopia lines
    # plt.plot([left_vd_point[0], left_iris_centroid[0]], [left_vd_point[1], left_iris_centroid[1]], color='black',linewidth=thickness)
    # plt.plot([right_vd_point[0], right_iris_centroid[0]], [right_vd_point[1], right_iris_centroid[1]], color='black',linewidth=thickness)
    
    # #  #make lists of rotated xy points for plotting 
    # fmx = [x[0] for x in fm]
    # fmy = [x[1] for x in fm]
    
    # lirx = [x[0] for x in left_iris]
    # liry = [x[1] for x in left_iris]
    
    # rirx = [x[0] for x in right_iris]  
    # riry = [x[1] for x in right_iris]  
    
    # #     #make a size list for scatter plot
    # sm = [1 for i in range(len(fmx))]
    # sr = [.1 for x in range(len(rirx))]
    # sl = [.1 for x in range(len(lirx))]
    
    # plt.scatter(left_upper_lid_x,left_upper_lid_y,l_u)
    # plt.scatter(left_lower_lid_x,left_lower_lid_y,l_l)
    # plt.scatter(right_upper_lid_x,right_upper_lid_y,r_u)
    # plt.scatter(right_lower_lid_x,right_lower_lid_y,r_l)
    # # plt.scatter(fmx, fmy, sm, c='white')
    # plt.scatter(lirx,liry, sl,c='black')
    # plt.scatter(rirx,riry, sr, c ='black')
    # plt.scatter(left_lower_brow_x, left_lower_brow_y,l2)
    # plt.scatter(right_lower_brow_x, right_lower_brow_y,r2)

    # plt.show()











