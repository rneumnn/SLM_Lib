import multiprocessing
import screeninfo
import numpy as np
import cv2
from time import sleep


class SLMScreen:
    """
    Class to handle the SLM-display. If used in a seperate function, take care of the point of return.
    This will lose you the SLMScreen object and causethe pipe to close which raises "BrokenPipeError: [WinError 109] The pipe has been ended".
    Concider a sleep(x) in the function and self.close() to prevent errors.
    Always use self.close() to close the windows. Everything else will lead to python crashing.
    """
    def __init__(self, screen = 1):
        self.screen= screeninfo.get_monitors()[screen]
        self.pixelWidthMM = self.screen.width_mm/self.screen.width
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.isOpen = False
        self._makeWindowSubprocess()
    
    def _makeWindowSubprocess(self):
        self.display_proc = multiprocessing.Process(
            target=SLMScreen.display_images,
            args=(self.child_conn, self.screen)
        )
        
    def _startWindowSubproccess(self):
        self.display_proc.start()
    
    def startShowing(self):
        self._startWindowSubproccess()
        self.isOpen = True


    @staticmethod
    def display_images(conn, screen:screeninfo.Monitor):
        window_name = "SLM Display"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, screen.x, screen.y)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            if conn.poll():
                img = conn.recv()
                if img is None:
                    break
                cv2.imshow(window_name, img)
                cv2.waitKey(1)
        cv2.destroyAllWindows()

    def updateImage(self, img: np.ndarray, s = 0):
        """
        Updates the image schown in the SLM window.

        Parameters
            :param img: Greyscale image with values [0;255]
            :type img: numpy.ndarray of shape(screen_x,screen_y)
        
        """
        if not self.isOpen:
            self.startShowing()
        self.parent_conn.send(img)
        sleep(s)

    def close(self):
        """
        Closes the SLM window.

        Parameters
        """
        self.parent_conn.send(None)
        self.display_proc.join()
        self.isOpen = False
    
    def restart(self):
        self._makeWindowSubprocess()