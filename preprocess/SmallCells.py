import math

class SmallCells:
    def __init__(self, h_latitude, l_latitude, h_longtitude, l_longtitude, radius):
        self.l_latitude = l_latitude
        self.l_longtitude = l_longtitude
        self.radius = radius

        self.x_num = int((h_longtitude-l_longtitude-0.5*radius)/(1.5*radius))+1
        self.h_longtitude = self.l_longtitude+(self.x_num-1)*1.5*radius+radius
        self.y_num = int((h_latitude-l_latitude)/(math.sqrt(3)*radius))
        self.h_latitude = self.l_latitude+self.y_num*math.sqrt(3)*radius

    def get_smallcell_index(self, no):
        if no <= self.x_num*self.y_num:
            y_index = int((no-1)/self.x_num)+1
            x_index = no-(y_index-1)*self.x_num
            return (x_index,y_index)
        else:
            y_index = self.y_num+1
            x_index = (no-self.x_num*self.y_num)*2
            return (x_index,y_index)

    def get_smallcell_coor(self, x_index, y_index):
        x_coor = (x_index-1)*1.5*self.radius + 0.5*self.radius
        if x_index % 2 == 0:
            y_coor = (y_index-1)*math.sqrt(3)*self.radius
        else:
            y_coor = (y_index-1)*math.sqrt(3)*self.radius + 0.5*math.sqrt(3)*self.radius
        return (x_coor,y_coor)

    def get_distance(self, coor1, coor2):
        temp1 = coor1[0]-coor2[0]
        temp2 = coor1[1]-coor2[1]
        return math.sqrt(temp1*temp1+temp2*temp2)

    def _get_smallcell_no(self, x_index, y_index):
        if y_index <= self.y_num:
            return (y_index-1)*self.x_num + x_index
        else:
            return self.x_num*self.y_num + int(x_index/2)

    def get_smallcell_no(self, latitude, longtitude):
        if latitude > self.h_latitude or latitude < self.l_latitude or longtitude > self.h_longtitude or longtitude < self.l_longtitude:
            return None

        lati = latitude - self.l_latitude
        longti = longtitude - self.l_longtitude

        lati_interval = math.sqrt(3)*self.radius
        longti_interval = 1.5*self.radius

        if longti < 0.5*self.radius:
            x_index = 1
            y_index = int(lati/lati_interval) + 1
            return (y_index-1)*self.x_num + 1
        else:
            x_index = int((longti - 0.5*self.radius)/longti_interval) + 2

        y_index = int(lati/lati_interval+0.5)
        candidates = []
        if x_index % 2 == 0:
            if y_index == 0:
                candidates.append((x_index-1,1))
                candidates.append((x_index,1))
            else:
                candidates.append((x_index-1,y_index))
                candidates.append((x_index-1,y_index+1))
                candidates.append((x_index,y_index+1))
        else:
            if y_index == 0:
                candidates.append((x_index-1,1))
                candidates.append((x_index,1))
            else:
                candidates.append((x_index-1,y_index+1))
                candidates.append((x_index,y_index))
                candidates.append((x_index,y_index+1))

        (x_index,y_index) = candidates[0]
        distance = self.get_distance((longti,lati),self.get_smallcell_coor(x_index,y_index))
        for i in range(1,len(candidates)):
            temp = self.get_distance((longti,lati),self.get_smallcell_coor(candidates[i][0],candidates[i][1]))
            if temp < distance:
                (x_index,y_index) = candidates[i]
                distance = temp

        return self._get_smallcell_no(x_index,y_index)

    def is_neighbor(self, sc1, sc2):
        sc1_index = self.get_smallcell_index(sc1)
        sc2_index = self.get_smallcell_index(sc2)
        if self.get_distance(self.get_smallcell_coor(sc1_index[0],sc1_index[1]), self.get_smallcell_coor(sc2_index[0],sc2_index[1])) < 1.2*math.sqrt(3)*self.radius:
            return True
        return False
