import numpy as np
def main():
    l = [0] * 500
    l_tot = [0] * 500
    total = 0
    
    for i in range(5):
        filename = 'experiment-2915276.' + str(i) + '.out'
        with open('./output/' + filename, 'r') as f:
            lines = f.readlines()
            
            
            for i, line in enumerate(lines):          
                if ("Epoch") in line:
                    
                    epoch = int(line.split(" ")[1])
                    
                    if (lines[i+2][0] == "0" or lines[i+2][0] == "1"):
                        l[epoch] += float(lines[i+2])
                        l_tot[epoch] += 1

    l = np.array(l)/l_tot[0]
    print(l)
    m = np.argmax(l)
    print(m)
    print(l[m])
    print(l[418])
    print(l[400])
    
        



if __name__ == "__main__":
    main()