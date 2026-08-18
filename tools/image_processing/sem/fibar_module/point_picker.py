# CODE for obtaining the starting locations from the image randomly #

import numpy as np

def point_picker(segmented_im, amount_of_points):
     """
     Picking points from the segmented image ("segmented_im"), when "amount_of_points" is found, the search stops
     
     """
     coords = []
     attempts = 0
     max_attempts = 5 

     while True:
          rnd_idx = np.random.randint(0, (len(np.where(segmented_im > 0)[0])), 1)
          
          x, y = np.where(segmented_im > 0)[0][rnd_idx][0], np.where(segmented_im > 0)[1][rnd_idx][0]

          attempts += 1

          #relative to px pos
          try:
               # the location needs to be at least one pixel away from all of the borders
               # U-upper, R-right, L - left, B - bottom 
               U = segmented_im[x-1][y]
               R = segmented_im[x][y+1]
               L = segmented_im[x][y-1]
               B = segmented_im[x+1][y]
          #if too much in the border - find a better location 
          except:
               continue

          # safe check - exclusive or 
          if (U == 0 and L == 0) ^ (U == 0 and R == 0) ^ (B == 0 and L == 0) ^ (B == 0 and R == 0):

               # ensuring that the quarters don't have the same amount of whites

               n = 13
               h, w = segmented_im.shape[:2]

               if (y+n) > w or (x+n) > h or (y-n) < 0 or (x-n) < 0:
                    n = np.min((abs(0-x), (h-x), (w-y), abs(0-y)))
               
               # #Creating a 13x13/nxn window where x,y is the midpoint 
               kernel_1 = segmented_im[x-(n//2):x+(n//2+1), y-(n//2):y+(n//2+1)]

               # # find biggest sum - direction
               UL = np.sum(kernel_1[:n//2, :n//2])
               UR = np.sum(kernel_1[:n//2, n//2+1:])
               LL = np.sum(kernel_1[n//2+1:, :n//2])
               LR = np.sum(kernel_1[n//2+1:, n//2+1:])


               # returning strings of quarter 
               quarters = np.array(["UL", "UR", "LL", "LR"])
               # https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum
               # "winners" is basically the direction where the closest white pixel might be
               winners = quarters[np.flatnonzero(np.array([UL, UR, LL, LR]) == np.max(np.array([UL, UR, LL, LR])))]

               if len(winners) == 2: 
                    if (np.array(["UL", "LR"]) == winners).all() or (np.array(["UR", "LL"]) == winners).all():
                         continue

               if len(winners) != 4:
                    attempts = 0

                    # uniqueness (was one pair at 1000 points)
                    if amount_of_points < 1000:
                         #print(x,y)
                         coords.append((x,y))
                    else:
                         if (x,y) not in coords:
                              coords.append((x,y))
                    
                    #print(len(coords))

               
               if attempts == max_attempts: 
                    break

               # amount of points chosen
               if len(coords) == amount_of_points:
                    break
          
     return coords
