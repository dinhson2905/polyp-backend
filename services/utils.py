import base64
import os
import time
import traceback
import datetime

def convert_base64_to_image(image_id, image_name, base64_code):
    try:
        imgdata = base64.b64decode(base64_code)
        if "." in image_name:
            ext = image_name.split(".")[-1]
        else:
            ext = "png"
        
        saved_image_name = str(int(time.time())) + "_" + image_id + "_" + image_name + "." + ext
        saved_image_filepath = os.path.join('static/inputimages', saved_image_name)
        
        print("Save from base64 to image: " + saved_image_filepath)
        with open(saved_image_filepath, 'wb') as f:
            base64_decoded = base64.b64decode(base64_code)
            f.write(base64_decoded)
            print("Saved image succeed!")
        return saved_image_filepath
    except Exception as e:
        traceback.print_exc()
        return None
