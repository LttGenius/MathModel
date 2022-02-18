import csv

def dataRead(filename:str):
#文件地址
    with open(filename) as f:
        return list(csv.reader(f))

if __name__ == '__main__':
    file=('C:/Users/mjl/Desktop/Math/2022_MCM_ICM_Problems/BCHAIN-MKPRU.csv','C:/Users/mjl/Desktop/Math/2022_MCM_ICM_Problems/LBMA-GOLD.csv')
    print(dataRead(file[0]))



