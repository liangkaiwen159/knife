def cross_product(point1, point2):
    return point1[0] * point2[1] - point1[1] * point2[0]

def is_on_seg(point, segment):
    p1, p2 = segment
    
    if min(p1[0],p2[0]) <= point[0] <= max(p1[0],p2[0]) \
        and min(p1[1],p2[1]) <= point[1] <= max(p1[1],p2[1]):
            return True
    else:
        return False

def segments_instance(segment1, segment2):
    p1, p2 = segment1
    p3, p4 = segment2

    v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    v2 = [p4[0] - p3[0], p4[1] - p3[1]]

    c1 = cross_product(v1, [p3[0] - p1[0], p3[1] - p1[1]])
    c2 = cross_product(v1, [p4[0] - p1[0], p4[1] - p1[1]])
    c3 = cross_product(v2, [p1[0] - p3[0], p1[1] - p3[1]])
    c4 = cross_product(v2, [p2[0] - p3[0], p2[1] - p3[1]])
    
    if c1 *c2 < 0 and c3*c4 <0:
        return True
    elif c1==0 and is_on_seg(p1, segment2):
        return True
    elif c2 ==0 and is_on_seg(p2,segment2):
        return True
    elif c3 ==0 and is_on_seg(p3, segment1):
        return True
    elif c4==0 and is_on_seg(p4,segment1):
        return True
    else:
        return False

seg1 = [(0,0),(3,3)]
seg2 = [(0,3),(3,0)]
print(segments_instance(seg1,seg2))
