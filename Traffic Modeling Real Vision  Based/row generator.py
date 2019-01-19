
counter = 0
vehicles = ["bicycle","motorcycle","passenger","passenger/van","truck","bus"]
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
    print("<vehicle id='"'{}'"' type='"'{}'"' route='"'{}'"' depart='"'{}'"'/>".format(vehcile_id,vehicles[vcounter],route_id,counter))
    rcounter += 1