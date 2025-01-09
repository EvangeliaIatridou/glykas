from matplotlib import pyplot as plt

with open('predictions.txt') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line=="\n":
            break
        line.strip()
        lineElements=line.split(',')
        x1=float(lineElements[0])
        x2=float(lineElements[1])
        category=int(lineElements[2])
        if category == 1:
            plt.plot(x1,x2,marker='+',color='magenta')
        elif category == 2:
            plt.plot(x1,x2,marker='+',color='green')
        elif category == 3:
            plt.plot(x1,x2,marker='+',color='blue')
        elif category == 4:
            plt.plot(x1,x2,marker='+',color='red')
        elif category == 0:
            plt.plot(x1,x2,marker='*',color='gold')

plt.show()
