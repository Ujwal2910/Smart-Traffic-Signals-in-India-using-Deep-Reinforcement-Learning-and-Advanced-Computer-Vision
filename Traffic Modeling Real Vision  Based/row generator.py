
counter = 0
vehicles = ["type1","type2","type3","type4","type5","type6"]
#routes = [10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44,130,131,132,133,134,140,141,142,143,144]
#routes = [1,4]
routes = [12,141]
vcounter = 0
rcounter = 0
r = 0

for i in range(4000):

    if i%5 == 0:
        counter +=5

    if i%10 == 0:
        vcounter += 1


    if vcounter-6 == 0:
        vcounter=0

    rcounter += 1
    if rcounter < 2000:
        r = 0
    else:
        r = 1


    vehcile_id = vehicles[vcounter]+str(int(i))
    route_id = "r"+str(routes[r])
    print("<vehicle id='"'{}'"' type='"'{}'"' route='"'{}'"' depart='"'{}'"'/>".format(i,vehicles[vcounter],route_id,counter))
    rcounter += 1