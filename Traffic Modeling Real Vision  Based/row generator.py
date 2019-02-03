
counter = 0
vehicles = ["type1","type2","type3","type4","type5","type6"]
#routes = [10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44,130,131,132,133,134,140,141,142,143,144]
routes = [4,"1_stop"]
#routes = [0,1,2,3,4,5,6,7,8,9,10,11]
vcounter = 0
rcounter = 0
ratio = 1
c=0
for i in range(2000):

    if i%5 == 0:
        counter +=5

    if i%10 == 0:
        vcounter += 1

    ##
    if vcounter-6 == 0:
        vcounter=0




    #ratio of vehicles in lane switching part

    # if i%666==0:
    #
    #     #666 is for 10 min change
    #     #333 is for 5 min change
    #     #66 is for 1 min change
    #     #1000 is for 15 min change
    #     c+=1
    # if c%2==0:
    #     if ratio >10:
    #         #10 is for 9:1 change
    #         #9 is for 8:2 change
    #         #8 is for 7:3 change
    #         rcounter = 1
    #         if ratio == 11:
    #             ratio = 1
    #     else:
    #         rcounter = 0
    # else:
    #     if ratio >10:
    #         rcounter = 0
    #         if ratio == 11:
    #             ratio = 1
    #
    #     else:
    #         rcounter = 1


    # if ratio >8:
    #     rcounter = 1
    #     if ratio ==11:
    #         ratio = 1
    # else:
    #     rcounter = 0

    # if rcounter-12 == 0:
    #     rcounter=0



    #obstacle introduction-
    if i%666==0:

        #666 is for 10 min change
        #333 is for 5 min change
        #66 is for 1 min change
        #1000 is for 15 min change
        c+=1
    if c%2==0:
        rcounter = 1

    else:
        rcounter = 0



    vehcile_id = vehicles[vcounter]+str(int(i))
    route_id = "r"+str(routes[rcounter])
    print("<vehicle id='"'{}'"' type='"'{}'"' route='"'{}'"' depart='"'{}'"'/>".format(i,vehicles[vcounter],route_id,counter))
    #rcounter += 1
    #ratio +=1