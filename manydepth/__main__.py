from . import manydepth
import matplotlib.pyplot as plt
import os
from .utils import git_dir
import subprocess

if __name__=="__main__":
    # Webcam depth
    import cv2
    cap = cv2.VideoCapture(0)

    # Clone the repo to access assets/test_sequence_intrinsics.json
    
    if not os.path.isdir(git_dir):
        command = "git clone https://github.com/AdityaNG/manydepth " + git_dir
        print(command)
        res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
        
    m = manydepth()

    plt.ion()
    # plt.draw()
    plt.show(block=False)

    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    old_ret, old_frame = ret, frame = cap.read()
    while(True):
        try:
            # Capture the video frame
            # by frame
            ret, frame = cap.read()

            #frame = cv2.imread('monodepth2/assets/test_image.jpg')
        
            # Display the resulting frame
            depth = m.eval(frame, old_frame)

            ax1.imshow(frame)
            ax2.imshow(depth)

            old_ret, old_frame = ret, old_frame
            
            #cv2.imwrite('tmps/frame.png', frame)
            #cv2.imwrite('tmps/depth.png', depth)
            #depth.save('depth.jpeg')
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            #time.sleep(0.01)
            plt.pause(0.001)
        except Exception as e:
            print(e)
            break
    
    plt.show()
    
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    #cv2.destroyAllWindows()