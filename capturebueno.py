import sys
import numpy as np
import pyzed.sl as sl
import cv2
import os

help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

def point_cloud_format_name(): 
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing") 
  
def depth_format_name(): 
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing") 

def get_image_count(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            count += 1
    return count

def save_image(image, folder):
    count = get_image_count(folder)
    image_id = count + 1
    cv2.imwrite(f"{folder}/{image_id}.jpg", image)
    print(f"Image saved as {folder}/{image_id}.jpg")

def save_point_cloud(zed, filename) :
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_depth(zed, filename) :
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)
    

def process_key_event(zed, key, objetos):
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext
    
    global saving_images
    global saving_dir
    
    if key == ord('s'):
        saving_images = not saving_images
        if saving_images:
            print("WARNING: Saving images")
        else:
            print("WARNING: Stoped saving images")
            
    if key in objetos:
        new_dir = objetos[key]
        if saving_dir != new_dir:
            saving_dir = new_dir
            print("WARNING: Changed saving directory to " + saving_dir)
    # if key == ord('b'):
    #     saving_dir = bowl_dir
    #     print("WARNING: Changed saving directory to " + saving_dir)
    
    # if key == ord('m'):
    #     saving_dir = milk_dir
    #     print("WARNING: Changed saving directory to " + saving_dir)
    
    # if key == ord('c'):
    #     saving_dir = cereal_dir
    #     print("WARNING: Changed saving directory to " + saving_dir) 

    # if key == 100 or key == 68:
    #     save_depth(zed, path + prefix_depth + str(count_save))
    #     count_save += 1
    # elif key == 110 or key == 78:
    #     mode_depth += 1
    #     depth_format_ext = depth_format_name()
    #     print("Depth format: ", depth_format_ext)
    # elif key == 112 or key == 80:
    #     save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
    #     count_save += 1
    # elif key == 109 or key == 77:
    #     mode_point_cloud += 1
    #     point_cloud_format_ext = point_cloud_format_name()
    #     print("Point Cloud format: ", point_cloud_format_ext)
    # elif key == 104 or key == 72:
    #     print(help_string)
    # elif key == 115:
    #     save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
    #     count_save += 1
    # else:
    #     a = 0

# dataset_breakfast/
# objetos = {
#     ord('d'): 'default',
#     ord('c'): 'cereal',
#     ord('m'): 'milk',
#     ord('e'): 'spoon',
#     ord('b'): 'bowl'
# }

# dataset_bolsas/
objetos = {
    ord('d'): 'default',
    ord('r'): 'rojo',
    ord('a'): 'amarillo',
    # ord('e'): 'spoon',
    # ord('b'): 'bowl'
}

def main() :

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # root_dir = "/home/oscar/Repositories/home/vision/dataset_breakfast"
    root_dir = "/home/oscar/Repositories/home/vision/dataset_bolsas"
    global saving_dir
    
    for key in objetos:
        if key == ord('s'):
            raise ValueError("The first character of the object name cant be 's' (used for saving)")

        objetos[key] = os.path.join(root_dir, objetos[key])
        os.makedirs(objetos[key], exist_ok=True)
    
    saving_dir = objetos[ord('d')]
    
    global saving_images
    saving_images = False
    
    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    key = ' '
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            depth_image_ocv = depth_image_zed.get_data()

            cv2.imshow("Image", image_ocv)
            # cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(1000 // 4)

            process_key_event(zed, key, objetos)            
            
            if saving_images:
                save_image(image_ocv, saving_dir)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()