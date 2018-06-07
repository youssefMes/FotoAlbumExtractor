import numpy as np 
import cv2

class DessinerLigne:
    def dessinerLigne(self):
        # Create a black image
        self.img=np.zeros((512,512,3),np.uint8)

        # Draw a diagonal blue line with thickness of 5 px
        cv2.line(self.img,(0,0),(511,511),(255,0,0),5)
        cv2.imshow("Image", self.img)
        # If q is pressed then exit program
        self.k=cv2.waitKey(0)
        if self.k==ord('q'):
            cv2.destroyAllWindows()

if __name__=="__main__":
    DL=DessinerLigne()
    DL.dessinerLigne()