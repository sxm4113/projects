import cv2
import numpy as np
import math
import matplotlib.pylab as plt
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.patches as mpatches

def display(image):
    cv2.imshow("input", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

        return np.isnan(y), lambda z: z.nonzero()[0]

class Mtf:

    def __init__(self, filename):

        img = cv2.imread(filename, 0)

        self.data = img[358:671, 1030:1109]

        _, th = cv2.threshold(self.data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        minv = np.amin(self.data)
        maxv = np.amax(self.data)
        self.threshold = th*(maxv - minv) + minv
        below_thresh = ((self.data >= minv) & (self.data <= self.threshold))
        above_thresh = ((self.data >= self.threshold) & (self.data <= maxv))
        area_below_thresh = self.data[below_thresh].sum()/below_thresh.sum()
        area_above_thresh = self.data[above_thresh].sum()/above_thresh.sum()

        self.threshold = (area_below_thresh - area_above_thresh)/2 + area_above_thresh

        edges = cv2.Canny(self.data, minv, maxv-5)
        plt.figure(1)
        plt.imshow(edges, cmap='gray')
        plt.title("Detected Edge")
        row_edge, col_edge = np.where(edges == 255)
        z = np.polyfit(np.flipud(col_edge), row_edge, 1)
        angle_radians = np.arctan(z[0])
        angle_deg = angle_radians * (180/3.14)
    
        print("angle_deg = ", angle_deg)
        if abs(angle_deg) < 45:
            self.data = np.transpose(self.data)
        
        self.compute_esf()
        
    def compute_esf(self):

        kernel = np.ones((3, 3), np.float32)/9
        smooth_img = cv2.filter2D(self.data, -1, kernel)
        self.data = cv2.GaussianBlur(self.data, (3, 3), 0)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        row = self.data.shape[0]
        column = self.data.shape[1]
        edge_range = 13
        array_values_near_edge = np.empty([row, edge_range])
        array_positions = np.empty([row, edge_range])
        edge_pos = np.empty(row)
        smooth_img = smooth_img.astype(float)

        for i in range(0, row):

            diff_img = smooth_img[i, 1:] - smooth_img[i, 0:(column-1)]
            abs_diff_img = np.absolute(diff_img)
            abs_diff_max = np.amax(abs_diff_img)
            if abs_diff_max == 1:
                raise IOError('No Edge Found')
            app_edge = np.where(abs_diff_img == abs_diff_max)
            bound_edge_left = app_edge[0][0] - 2
            bound_edge_right = app_edge[0][0] + 3
            strip_cropped = self.data[i, bound_edge_left:bound_edge_right].astype(float)

            temp_y = np.arange(1, 6)
            f = interpolate.interp1d(strip_cropped, temp_y,fill_value="extrapolate")

            edge_pos_temp = f(self.threshold)


            edge_pos[i] = edge_pos_temp + bound_edge_left - 1
            bound_edge_left_expand = app_edge[0][0] - int(np.floor(edge_range//2) )
            bound_edge_right_expand = app_edge[0][0] + int(np.floor(edge_range//2) +1)
            array_values_near_edge[i, :] = self.data[i, bound_edge_left_expand:bound_edge_right_expand]
            array_positions[i, :] = np.arange(bound_edge_left_expand, bound_edge_right_expand)

        y = np.arange(0, row)
        nans, x = nan_helper(edge_pos)
        edge_pos[nans] = np.interp(x(nans), x(~nans), edge_pos[~nans])

        array_positions_by_edge = array_positions - np.transpose(edge_pos * np.ones((edge_range, 1)))
        num_row = array_positions_by_edge.shape[0]
        num_col = array_positions_by_edge.shape[1]
        array_values_by_edge = np.reshape(array_values_near_edge, num_row*num_col, order='F')
        array_positions_by_edge = np.reshape(array_positions_by_edge, num_row*num_col, order='F')

        bin_pad = 0.0001
        pixel_subdiv = 0.10
        topedge = np.amax(array_positions_by_edge) + bin_pad + pixel_subdiv
        botedge = np.amin(array_positions_by_edge) - bin_pad
        binedges = np.arange(botedge, topedge+1, pixel_subdiv)

        numbins = np.shape(binedges)[0] - 1

        binpositions = binedges[0:numbins] + (0.5) * pixel_subdiv

        whichbin = np.digitize(array_positions_by_edge, binedges)
        binmean = np.empty(numbins)

        for i in range(0, numbins):
            flagbinmembers = (whichbin == i)
            binmembers = array_values_by_edge[flagbinmembers]

            binmean[i] = np.mean(binmembers)
        nans, x = nan_helper(binmean)
        t = binmean.copy()

        binmean[nans] = np.interp(x(nans), x(~nans), binmean[~nans])
        self.esf = binmean
        self.xesf = binpositions
        self.xesf = self.xesf - np.amin(self.xesf)

        self.esf_smooth = savgol_filter(self.esf, 51, 3)

        plt.figure(2)
        plt.title("ESF Curve")
        plt.xlabel("pixel")
        plt.ylabel("DN Value")

        plt.plot(self.xesf, self.esf)
        yellow_patch = mpatches.Patch(color='yellow', label='Raw ESF')

        self.compute_lsf()

    def compute_lsf(self):
    
        diff_esf = abs(self.esf[1:] - self.esf[0:(self.esf.shape[0] - 1)])
        diff_esf = np.append(0, diff_esf)
        self.lsf = diff_esf
        diff_esf_smooth = abs(self.esf_smooth[0:(self.esf.shape[0] - 1)] - self.esf_smooth[1:])
        diff_esf_smooth = np.append(0, diff_esf_smooth)
        self.lsf_smooth = diff_esf_smooth

        self.compute_mtf()
        
    def compute_mtf(self):
        fft_dimention = 2048
        mtf = np.absolute(np.fft.fft(self.lsf, fft_dimention))
        mtf_smooth = np.absolute(np.fft.fft(self.lsf_smooth, fft_dimention))
        mtf_final = np.fft.fftshift(mtf)
        mtf_final_smooth = np.fft.fftshift(mtf_smooth)

        display_range = 127

        x_mtf_final = np.arange(0,1,1./display_range)*100

        mtf_final = mtf_final[fft_dimention//2:fft_dimention//2+display_range]/np.amax(mtf_final[fft_dimention//2:fft_dimention//2+display_range])

        plt.figure(3)
        plt.plot(x_mtf_final, mtf_final)
        plt.xlabel("cycles/mm")
        plt.ylabel("Modulation Factor")
        plt.title("MTF Curve")

        print ("MTF50 = ",np.interp(0.5, mtf_final[::-1],x_mtf_final[::-1]) ,"Cycle/mm")
        print ("MTF @ 15 Cycle/mm = ",np.interp(15.0, x_mtf_final,mtf_final))
        print ("MTF @ 30 Cycle/mm = ",np.interp(30.0, x_mtf_final,mtf_final))

        plt.show()
    
if __name__ == '__main__':
    Mtf('./input_image/nfov_e_vert.bmp')
       