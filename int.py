import numpy as np

f_ptr ='./data/train.csv'
data = open(f_ptr).readlines()

seq = {} #empty dic
for i in range(1,len(data)):
    line = data[i].replace('"','').split(',')
    seq[int(line[0])]=[int(i) for i in line[1:]]


def coeff_rec(seq, order):
    minlength = 7
    if len(seq)< max((2*order+1), minlength):
                return None
    A,b=[],[]
    for i in range(order):
        A.append(seq[i:i+order])
        b.append(seq[i+order])
    # Coeff
    A,b = np.array(A),np.array(b)
    try:
        if np.linalg.det(A) == 0:
            return None
    except TypeError:
        return None
    coeff = np.linalg.inv(A).dot(b)
    
    for i in range(2*order,len(seq)):
        pred = np.sum(coeff*np.array(seq[i-order:i]))
        if abs(pred - seq[i]) > 1e-2:
            return None
    return list(coeff)


def predict_next(seq,coeffs):
    order = len(coeffs)
    predict = np.sum(coeffs*np.array(seq[-order:]))
    return int(round(predict))


order2Seq = {}
for i in seq:
    coeff = coeff_rec(seq[i],2)
    if coeff != None:
        predict = predict_next(seq[i],coeff)
        order2Seq[i] = (predict,coeff) 
print ("We found %d sequences\n" %len(order2Seq))
print ("ID,  Prediction,  Coefficients")
for key in sorted(order2Seq)[0:10]:
    print(f"{key}, {order2Seq[key][0]}, {[round(i) for i in order2Seq[key][1]]}")     

