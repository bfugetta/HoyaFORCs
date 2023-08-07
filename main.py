import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from math import *
from scipy.optimize import curve_fit
import struct
import time

nonstandard = True #use the nonstandard CNN


########### Changing will vary results
random.seed(690)
np.random.seed(690)

########## Hyperparameters used to train the neural network
mu = 0.9     #momentum
wd = 0       #weight decay
rate = 0.01  #learning rate


parameter = 4 #DMI

torch.autograd.set_detect_anomaly(True)

train_loader = 0

bins = 11 #the number of output nodes used in the nonstandard CNN

z_low = 10e-9
z_up = 100e-9

phi_low = 0
phi_up = 360

D_low = 0
D_up = 5e-3

A_low = 1e-12
A_up = 3.5e-11

M_low = 200e3
M_up = 1400e3

T_c_up = ((120e-9/16)*A_up)/1.380649e-23
T_c_low = ((120e-9/16)*A_low)/1.380649e-23  #simple upper and lower estimates for critical temperature

T_low = T_c_low/4 
T_up = T_c_up + 3*T_c_low/4



#location of the data
label_data_root = "./labels.bin"
image_data_root = "./images.bin"


#These are the locations you wish to store the results of the training data, should be .txt files
filename_test = ''
filename_train = ''
best_test = ''
best_train = ''
#Wherever you want the final weight matrix to be saved, should be .pt file
end_model_save_root = ''

assert end_model_save_root != filename_test != filename_train != best_test != best_train != ""


def decode_3D(root, endianness = 'big'):
    types = { #important information about different numerical datatypes including their code for being unpacked by the struct module
    8: (torch.uint8, np.uint8, np.uint8, 'B', 1),
    9: (torch.int8, np.int8, np.int8, 'b', 1),
    11: (torch.int16, np.dtype('>i2'), 'i2', 'H', 2),
    12: (torch.int32, np.dtype('>i4'), 'i4', 'I', 4),
    13: (torch.float32, np.dtype('>f4'), 'f4', 'f', 4),
    14: (torch.float64, np.dtype('>f8'), 'f8', 'd', 8)
    }
    
    if endianness == 'big':
        end = '>'
    if endianness == 'little':
        end = '<'
    if endianness == 'native':
        end = '='
    
    f = open(root, 'rb')
    
    data = f.read() #turns binary file into an array of bytes
    
    f.close() #don't wanna leave that open
    
    nd = data[3] #the number of dimensions of the data in the file
    ty = data[2] #the type of numbers stored in the file
    assert nd == 3
    assert ty == 8
    m = types[ty] #relevant information about the types of numbers
    offset = 4 * (nd + 1) #byte location where the first real data will be stored
    s = list(struct.unpack(end + 'I'*nd, data[4: offset])) #list of the number of entries in each of the dimensions
    dataset = []
    
    for i in range(s[0]): #loops through the first dimension (images)
        image = []
        for j in range(s[1]): #loops through the second dimension (rows)
            image.append(list(struct.unpack(end + m[3]*s[2], data[offset + s[2]* j      + s[1]*s[2]*i :
                                                                  offset + s[2]*(j + 1) + s[1]*s[2]*i]))) #unpacks the third dimension (pixels) and adds the row of pixels to the image being built
        dataset.append(image) #adds the complete image to the final dataset
    
    return dataset #give us the full dataset stored at root as a list of lists


def decode_2D(root, endianness = 'big'):
    types = { #important information about different numerical datatypes like including their code for being unpacked by the struct module
    8: (torch.uint8, np.uint8, np.uint8, 'B', 1),
    9: (torch.int8, np.int8, np.int8, 'b', 1),
    11: (torch.int16, np.dtype('>i2'), 'i2', 'H', 2),
    12: (torch.int32, np.dtype('>i4'), 'i4', 'I', 4),
    13: (torch.float32, np.dtype('>f4'), 'f4', 'f', 4),
    14: (torch.float64, np.dtype('>f8'), 'f8', 'd', 8)
    }
    
    if endianness == 'big':
        end = '>'
    if endianness == 'little':
        end = '<'
    if endianness == 'native':
        end = '='
    
    f = open(root, 'rb')
    
    data = f.read() #turns binary file into an array of bytes
    
    f.close() #don't wanna leave that open

    nd = data[3] #the number of dimensions of the data in the file
    ty = data[2] #the type of numbers stored in the file
    assert data[0] == 0
    assert data[1] == 0 #make sure we have the right file
    assert nd == 2 #make sure we called the right function
    assert ty >= 8 and ty <= 14 #ensure its a valid numerical type for this function
    m = types[ty] #relevant information about the types of numbers
    offset = (4 * (nd + 1)) #byte location where the first real data will be stored
    s = list(struct.unpack(end + 'I'*nd, data[4: offset])) #list of the number of entries in each of the dimensions
    dataset = []
    
    for i in range(s[0]): #loops through the first dimension (labels)
        dataset.append(list(struct.unpack(end + m[3]*s[1], data[offset + m[4] * s[1] *  i :
                                                                offset + m[4] * s[1] * (i + 1)]))) #unpacks the second dimension (parameters) and adds the label of parameters to the final dataset
    
    return dataset #give us the full dataset stored at root as a list of lists


train_images = decode_3D(image_data_root)
train_labels = decode_2D(label_data_root)
train_labels = np.array(train_labels)

ups = [M_up, T_up, A_up, max(train_labels[..., 3]), D_up, z_up, max(max(train_labels[..., 6]), abs(min(train_labels[..., 6]))), phi_up] #the upper bounds of the K_u and theta values must be determined by the data itself because they were generated with gaussians instead of uniform distributions
lows = [M_low, T_low, A_low, 0, D_low, z_low, -ups[6], phi_low] #the lower bound of K_u can be set to 0, but the theta lower bound is just the negative of the upper bound

for i in range(len(lows)): #normalize the labels to between 0 and 10 using their respective lower and upper bounds
    train_labels[..., i] -= lows[i]
    train_labels[..., i] /= (ups[i] - lows[i])/(bins - 1)
train_labels = train_labels.tolist()

train_dataset = []
test_dataset = []
for i in range(len(train_images)):
    r = random.random()
    data = (
             torch.cat( #takes in a tuple of tensors
                           ( 
                               torch.tensor([[train_images[i]]]),             #we need an extra two dimensions, the first so that we can grab the (FORC image, M_s image) pair for every datapoint, and the second so we can grab the set of FORC images accross all datapoints the set of M_s images accross all datapoints
                               torch.ones(1, 1, 61, 61)*(train_labels[i][0])  #this concatenation is a sneaky way to subvert issues of combining data of different types, we can easily convert the normalized M_s into an image, combine it with the FORC images of the same dimension, and extract the M_s later
                           )
                      ),                                #this concatenated tensor is the input information for the CNN, which is the first entry in the tuple called data 
             torch.tensor(train_labels[i][parameter])   #this tensor is the correct answer the CNN is trying to guess, which is the second entry in the tuple called data
            )
    if r <= 0.8: #randomly choose around 80% of the full dataset
        train_dataset.append(data) #the dataset ends up being a list of tuples of the form (input, target), where input is itself a 4-D tensor with the first dimension separating FORC from M_s the second dimension containing the single image the third dimension containing the 61 rows and the fourth dimension containing the 61 pixels
    else:
        test_dataset.append(data)  #same as above





def parabolic(x, a, b, c): #used for curve fitting
    return a*(x**2) + b*x + c

def find_gaussian_max(output): #just a shortcut so we don't have to recode all that we did in the other functions
    params, covar = curve_fit(parabolic, [i for i in range(len(output))], output)
    maximum = -params[1]/(2*params[0])
    return maximum

def find_new_gaussian_max(output): #this is for if we only have a single output instead of a tensor of them
    maximum = output.argmax()
    output = output.tolist()
    length = len(output)
    if maximum == 0 or maximum == 1:
        put = [output[0], output[1], output[2], output[3], output[4]]
        guess = (find_gaussian_max(put) + length) - length
    elif maximum == length - 1 or maximum == length - 2:
        put = [output[length - 5], output[length - 4], output[length - 3], output[length - 2], output[length - 1]]
        guess = find_gaussian_max(put) + length - 5
    else:
        put = [output[maximum - 2], output[maximum - 1], output[maximum], output[maximum + 1], output[maximum + 2]]
        guess = find_gaussian_max(put) + maximum - 2
    return guess


def find_new_gaussian_max_tensor(outputs): #get the maximum value from a tensor of many 11-node outputs
    maxima = outputs.argmax(dim = 1)
    sloice = torch.clamp(maxima, 2, 8) #don't allow the maximum index be too near the boundaries
    slices = torch.stack([torch.tensor(i) for i in zip(sloice - 2, sloice + 2)]) #find which indices will be needed from each output
    length = len(outputs[0])
    outs = torch.stack([outputs[i][slices[i][0].item():slices[i][1].item() + 1] for i in range(len(slices))]) #grab the 5 nodes nearest the maximum
    n = len(outs[0])
    x = torch.tensor([i for i in range(n)])
    x = x.to(torch.device("cuda")) #needed for batch operation
    x2 = torch.tensor([i**2 for i in range(n)])
    x2 = x2.to(torch.device("cuda"))
    x3 = torch.tensor([i**3 for i in range(n)])
    x4 = torch.tensor([i**4 for i in range(n)])
    x_bar = x.float().mean()
    x_2bar = x2.float().mean()
    x_3bar = x3.float().mean()
    x_4bar = x4.float().mean()
    y_bars = outs.mean(dim = 1)
    ####this is a bunch of established math to find the parabola of best fit, you don't have to understand it, but it finds the x-value of the maximum of the parabola that best fits the slice of the output we grabbed above
    xys = torch.mul(outs, x)
    xy_bars = xys.mean(dim = 1)
    x2ys = torch.mul(outs, x2)
    x2y_bars = x2ys.mean(dim = 1)
    cs = ((x2y_bars - x_2bar*y_bars)*(x_2bar - x_bar**2) - (xy_bars - x_bar*y_bars)*(x_3bar - x_bar*x_2bar))/((x_4bar - x_2bar**2)*(x_2bar - x_bar**2) - (x_3bar - x_bar*x_2bar)**2)
    i_guesses = -0.5*(xy_bars - x_bar*y_bars - (x_3bar - x_bar*x_2bar)*cs)/((x_2bar - x_bar**2)*cs)
    guesses = i_guesses + sloice - 2
    
    return guesses
    


def meanvalue(iterable): #standard mean value calculation
    global device
    total = 0
    number = 0
    try:
        iterable = iterable.to(device)
    except:
        pass
    for i in iterable:
        total += i
        number += 1
    mean = total / number
    return mean

def sigmacalc(iterable): #standard standard deviation value calculation
    total = 1e-12
    number = 0
    average = meanvalue(iterable)

    for i in iterable:
        total += (i - average) ** 2
        number += 1
    sigma = sqrt(total / (number - 1))
    return sigma

def pearson_correlation(x, y): #standard pearson correlation calculation
    n = len(x)
    assert len(y) == n
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    x_bar = meanvalue(x)
    y_bar = meanvalue(y)
    s_x = sigmacalc(x)
    s_y = sigmacalc(y)


    total = 0

    for i in range(n):
        var_x = (x[i] - x_bar) / s_x
        var_y = (y[i] - y_bar) / s_y
        total += var_x * var_y

    return float(total / (n - 1))

def avg_abs_err(x, y): #standard <|error|> calculatoin
    n = len(x)
    assert len(y) == n
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    

    total_err = 0

    for i in range(n):
        total_err += abs(x[i] - y[i])

    return float(total_err / n)


def linfit(x, y): #standard equations for this
    x_bar = meanvalue(x)
    y_bar = meanvalue(y)
    s_xy = 0
    s_xx = 0
    n = len(x)
    assert len(y) == n

    for i in range(n):
        s_xy += (x[i] - x_bar) * (y[i] - y_bar)
        s_xx += (x[i] - x_bar) ** 2

    m = s_xy / s_xx  

    b = y_bar - m * x_bar

    return m, b


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        targ_length = len(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_1 = model(data) #tensor of all the results of the inputs being fed through the CNN
        
        
        if nonstandard:
            target = (20-((torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]*targ_length).to(device) - target.reshape([targ_length, 1]))**2)/2) #natural log of a gaussian with mu = target and sigma = 1 scaled by e^20, essentially creating a new target for the output
            loss_1 = F.mse_loss(output_1.float(), (target).float()) #standard loss function between the output and the log of the gaussian
        else:
            loss_1 = F.mse_loss(output_1, target.reshape_as(output_1).float())
        
        
        loss_1.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print("Batch ", batch_idx, "'s sample output: ", output_1[0])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_1.item()))



def test(model, device, test_loader, epoch, jk = True):
    global train_loader #only used to tell whether we have used this function on the training set or not
    model.eval()
    worst = 0
    preds = []
    targs = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            target = target.float()
            output_1 = model(data)
            
            
            if nonstandard:
                pred_1 = find_new_gaussian_max_tensor(output_1) #not simple
            else:
                pred_1 = output_1[..., 0] #simple
            
            
            targs.append(target)
            preds.append(pred_1)

            batch_worst = max(abs(pred_1 - target))
            if batch_worst > worst:
                worst = batch_worst

    targs = torch.cat(targs)
    preds = torch.cat(preds)

    correlation = pearson_correlation(targs, preds)
    abs_error = avg_abs_err(targs, preds)
    slope, intercept = linfit(targs, preds)
    
    if test_loader == train_loader: #are we testing the test set or the training set?
        if epoch == 0: #check if we need to create the file first
            with open(filename_train, 'w') as f:
                string = str(epoch)
                string += '\t' + str(float(abs_error))
                string += '\t' + str(float(slope))
                string += '\t' + str(float(correlation)) + '\n'
                f.writelines(string)
                print("epoch 0:", abs_error, slope, correlation, worst)
        else:
            with open(filename_train, 'a') as f:
                string = str(epoch)
                string += '\t' + str(float(abs_error))
                string += '\t' + str(float(slope))
                string += '\t' + str(float(correlation)) + '\n'
                f.writelines(string)
                print("epoch " + str(epoch) + ":", abs_error, slope, correlation, worst)
    else:
        if epoch == 0: #check if we need to create the file first
            with open(filename_test, 'w') as f:
                string = str(epoch)
                string += '\t' + str(float(abs_error))
                string += '\t' + str(float(slope))
                string += '\t' + str(float(correlation)) + '\n'
                f.writelines(string)
                print("epoch 0:", abs_error, slope, correlation, worst)
        else:
            with open(filename_test, 'a') as f:
                string = str(epoch)
                string += '\t' + str(float(abs_error))
                string += '\t' + str(float(slope))
                string += '\t' + str(float(correlation)) + '\n'
                f.writelines(string)
                print("epoch "+ str(epoch) + ":", abs_error, slope, correlation, worst)
    return [correlation, abs_error, slope, intercept, worst]



def correlate(model, device, test_loader, epoch, fn_train, fn_test):
    global train_loader #only used to tell whether we have used this function on the training set or not
    model.eval()
    targs = []
    preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.float()
            output_1 = model(data)
            for i in range(len(target)):

                
                if nonstandard:
                    pred_1 = find_new_gaussian_max(output_1[i])
                else:
                    pred_1 = output_1[i][0]


                preds.append(pred_1)
                targs.append(target[i])
                
                if i == 0:
                    print(pred_1, target[i])
    
    
    if test_loader == train_loader: #are we checking the test set or the training set?
        with open(fn_train, 'a') as f:
            for i in range(len(targs)):
                string = str(float(preds[i])) + '\t' + str(float(targs[i])) + '\n'
                f.writelines(string)
    else:
        with open(fn_test, 'a') as f:
            for i in range(len(targs)):
                string = str(float(preds[i])) + '\t' + str(float(targs[i])) + '\n'
                f.writelines(string)
    return 'done'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(8, 16, 3, stride = 1, padding = 0)
        self.dropout1 = nn.Dropout(0)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(((61 - 4)**2)*16 + 576 , 4096) #input images are 61x61, and each convolution removes the edge pixels reducing the overall image dimensions by 2 in each direction, so final feature maps are (61 - 4)x(61 - 4), and there are 16 of them from conv2. Additionally, there are the 576 nodes coming from the input M_s
        if nonstandard:
            self.fc2 = nn.Linear(4096, bins)
        else:
            self.fc2 = nn.Linear(4096, 501)
            self.fc3 = nn.Linear(501, 1)

    def forward(self, x):
        x   = x.float()
        
        #this is where our extra dimension from making the dataset is useful as it makes the next part possible as a batch operation
        img = x[::, 0] #61x61 image of FORC simulation
        lbl = x[::, 1] #61x61 "image" with pixel values equal to the normalized M_s
        
        
        img = self.conv1(img) #61x61 image becomes 8 59x59 feature maps
        img = F.relu(img) 
        img = self.dropout1(img) #no actual dropout because it is 0
        img = self.conv2(img) #8 59x59 feature maps become 16 57x57 feature maps
        img = F.relu(img)
        img = self.dropout2(img)
        img = torch.flatten(img, 1) #16 57x57 feature maps become a single layer with 51984 nodes
        
        #this section turns the M_s into a section of nodes to be appended to the output of conv2, the number of nodes was tested and 576 was chosen because it was equivalent to a maxpool with a kernel size of 2.5, which performed better than one of size 2 (resulting in 900 nodes) and one of size 3 (resulting in 400)
        lbl = F.max_pool2d(lbl, 13, stride = 1) #1 61x61 image becomes 1 49x49 image
        lbl = F.max_pool2d(lbl, 2) #1 49x49 image becomes 1 24x24 image
        lbl = torch.flatten(lbl, 1) #1 24x24 image becomes a single layer with 576 nodes
        
        
        x   = torch.cat((img, lbl), dim = 1) #combine the layer with 51984 nodes and the layer with 576 nodes into a single layer with 52560
        x   = self.fc1(x) #layer with 52560 nodes becomes a layer with 4096 nodes
        x   = F.relu(x)
        x   = self.dropout3(x)
        x   = self.fc2(x)
        if not nonstandard:
            x   = F.relu(x)
            x   = self.dropout4(x)
            x   = self.fc3(x)
        output = x.float() 
        return output


def main():
    global train_loader
    global device
    # Training settings
    parser = argparse.ArgumentParser(description = 'DMI from FORC images example')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--test-batch-size', type=int, default=5000, metavar='N',
                        help='input batch size for testing (default: 5000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    model = Net().to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr = rate, rho = mu, weight_decay = wd)

    scheduler = StepLR(optimizer, step_size=1, gamma=1)
    
    train_accuracy = test(model, device, train_loader, 0) #dummy variables so that the function can do the writing
    test_accuracy = test(model, device, test_loader, 0)
    
    
    
    epochs_without_improvement = 0
    epoch = 1
    best_stats = [4, 0, 0, 0] #[absolute average error, slope, correlation, performance], default: [1.8, 0, 0.6, 0]
    while epoch <= args.epochs:
        t_i = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        
        if epoch >= 200:
            epochs_without_improvement += 1 #only actually tick up if we are past 200 epochs
        
        accuracy = test(model, device, train_loader, epoch)
        accuracy = test(model, device,  test_loader, epoch)
        
        performance = 100*(best_stats[0] - accuracy[1]) + 25*(accuracy[2] - best_stats[1]) + 100*(accuracy[0] - best_stats[2]) #measure of the performace based on 3 important stats
        
        if ((best_stats[0] - accuracy[1]) >= 0) and ((accuracy[2] - best_stats[1]) >= 0) and ((accuracy[0] - best_stats[2]) >= 0) and (performance >= best_stats[3]) and epoch > 300: #only rewrite if we are better than our default stats and are passed epoch 300
            with open(filename_train, 'r') as f:
                g =  open(best_train, 'w')
                for line in f:
                    g.write(line)
                g.close()
            with open(filename_test, 'r') as f:
                g =  open(best_test, 'w')
                for line in f:
                    g.write(line)
                g.close()
            print(correlate(model, device, train_loader, epoch, best_train, best_test), ' train') #save the training stats for this best performance
            print(correlate(model, device,  test_loader, epoch, best_train, best_test),  ' test') #save the  testing stats for this best performance
            torch.save(model.state_dict(), end_model_save_root) #save the state of the CNN for this best performance
            best_stats = [4, 0, 0, performance] #[1.8, 0, 0.6, performance]
            epochs_without_improvement = 0 #we improved
        scheduler.step() #finished one epoch
        t_f = time.time()
        print('Time taken: ', t_f - t_i) #keep track on time it takes to run an epoch
        if epochs_without_improvement >= 200:
            epoch = args.epochs #exit the loop
        epoch += 1
    epoch = args.epochs
    print(correlate(model, device, train_loader, epoch, filename_train, filename_test), ' train') #need to log the final epoch's stats too
    print(correlate(model, device,  test_loader, epoch, filename_train, filename_test),  ' test')



if __name__ == '__main__':
    main()


