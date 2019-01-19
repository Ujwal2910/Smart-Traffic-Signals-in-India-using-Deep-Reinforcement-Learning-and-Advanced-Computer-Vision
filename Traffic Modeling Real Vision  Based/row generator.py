
counter = 0
vehicles = ["type1","type2","type3","type4","type5","type6"]
vcounter = 0
rcounter = 0

for i in range(200):

    if i%5 == 0:
        counter +=5

    if i%10 == 0:
        vcounter += 1


    if vcounter-6 == 0:
        vcounter=0

    if rcounter-12 == 0:
        rcounter=0


    vehcile_id = vehicles[vcounter]+str(int(i))
    route_id = "r"+str(rcounter)
    print("<vehicle id='"'{}'"' type='"'{}'"' route='"'{}'"' depart='"'{}'"'/>".format(i,vehicles[vcounter],route_id,counter))
    rcounter += 1