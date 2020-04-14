with open("datasets/tr_data4.txt", "r") as f:
    lines = f.readlines()
cnt_f = 0
cnt_m = 0
for line in lines:
    _, _, sex = line.split()
    if sex == "m":
        cnt_m+=1
    elif sex == "f":
        cnt_f+=1
print("male:{}".format(cnt_m))
print("female:{}".format(cnt_f))