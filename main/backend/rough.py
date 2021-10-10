linear = 0
for i in range(0, 16, 4):
    for j in range(i, i + (7 * 16), 16):
        for k in range(j, j + 4):
            if (k < 100):
                print(k)
            else:
                break
