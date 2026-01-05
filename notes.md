# Assignment 1

# Assignment 2 (CNNs)

## Development

1. Tried LeNet5 -> 5 % after 5 epochs. No learning
2. Tried Conv(16,5) - Conv(16,3) - MaxPool(2,2) - Dropout(0.3) - Con(32,3) - Con(32,3) - MaxPool(2,2) - Dropout(0.4) - fc - fc - fc | -> 20%
3. Remove last fc -> 40%
4. Switch out channels in beginning to 32 and add batchnorm -> `Conv(32,5, padd=1) - Conv(32,3) - BatchNorm - MaxPool(2,2) - Dropout(0.3) - Con(16,3, padd=1) - Con(16,3) - BatchNorm - MaxPool(2,2) - Dropout(0.4) - fc(x, out=120) - fc` -> 5 epochs -> 60%
5. Switch to AvgPool on first -> 57,4% (almost no learning from epoch 4 to 5)
6. Back to MaxPool + SDG Momentum to 0.9 + GELU as activation function. Epoch 1: 42%, Epoch 5: 54,8 %
7. Remove dropout and normalization to reach 65 at first
#### Model
   ```python
   Model(
  (conv_seq1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
    (1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): ReLU()
  )
  (conv_seq2): Sequential(
    (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): ReLU()
  )
  (fc1): Linear(in_features=256, out_features=100, bias=True)
  (fc2): Linear(in_features=100, out_features=10, bias=True)
)
   ```
#### Results

- Epoch 1: Train loss: 2.019977260840984, Validation loss 2.0153698921203613
- Accuracy on the validation set: 23.51 %, Accuracy on the validation set: 20.34 %
- Epoch 2: Train loss: 1.985371518424895, Validation loss 2.022616147994995
- Accuracy on the validation set: 22.95 %, Accuracy on the validation set: 21.64 %
- Epoch 3: Train loss: 1.9811604913045258, Validation loss 2.0308918952941895
- Accuracy on the validation set: 23.538 %, Accuracy on the validation set: 23.18 %
- Epoch 4: Train loss: 2.0033442659860077, Validation loss 2.0026535987854004
- Accuracy on the validation set: 22.388 %, Accuracy on the validation set: 23.4 %
- Epoch 5: Train loss: 1.9762487347432611, Validation loss 1.9836575984954834
- Accuracy on the validation set: 22.8 %, Accuracy on the validation set: 23.08 %

8. Removed chaining of conv instead conv - pool - act, Tried to keep the last out dimension not too low, and remove momentum from SGD

#### Model

**!! SO FAR BEST !!**

```python
Model(
  (conv_seq): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU()
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (5): ReLU()
    (6): Conv2d(32, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (8): ReLU()
  )
  (fc1): Linear(in_features=320, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=10, bias=True)
)
(20, 4, 4)
```

#### Results

- Epoch 1: Train loss: 1.8730545410801795, Validation loss 1.6607660055160522
- Accuracy on the training set: 31.528 %, Accuracy on the validation set: 36.5 %
- Epoch 2: Train loss: 1.4391990289120664, Validation loss 1.498759388923645
- Accuracy on the training set: 48.24 %, Accuracy on the validation set: 45.8 %
- Epoch 3: Train loss: 1.2623423415349984, Validation loss 1.2842403650283813
- Accuracy on the training set: 54.852 %, Accuracy on the validation set: 53.58 %
- Epoch 4: Train loss: 1.1445784967294963, Validation loss 1.1406561136245728
- Accuracy on the training set: 59.346 %, Accuracy on the validation set: 59.68 %
- Epoch 5: Train loss: 1.0557349602610198, Validation loss 1.071328043937683
- Accuracy on the training set: 62.584 %, Accuracy on the validation set: 62.08 %

9. Change on hyperparameter batch_size. Now 48

#### Results

- Epoch 1: Train loss: 1.7576432476727075, Validation loss 1.5666288137435913
- Accuracy on the training set: 36.0 %, Accuracy on the validation set: 40.76 %
- Epoch 2: Train loss: 1.363566156121606, Validation loss 1.3706778287887573
- Accuracy on the training set: 51.264 %, Accuracy on the validation set: 50.04 %
- Epoch 3: Train loss: 1.2095863186847835, Validation loss 1.1905837059020996
- Accuracy on the training set: 56.916 %, Accuracy on the validation set: 57.16 %
- Epoch 4: Train loss: 1.0948371397175243, Validation loss 1.1283273696899414
- Accuracy on the training set: 61.202 %, Accuracy on the validation set: 60.76 %
- Epoch 5: Train loss: 0.9990690208099168, Validation loss 1.0886411666870117
- Accuracy on the training set: 64.806 %, Accuracy on the validation set: 61.6 %

Faster training but seems like overfitting

10. Switch to max pooling on last layer:

**New best model so far**

### Model

```python
class Model(nn.Module):
    def __init__(self, channels=3, height=32, width=32):
        super().__init__()
        
        conv1 = nn.Conv2d(channels, 32, kernel_size=5, padding=2)
        h_out, w_out = out_dimensions(conv1, height, width)
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        h_out, w_out = h_out // 2, w_out // 2
        conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        h_out, w_out = out_dimensions(conv2, h_out, w_out)
        pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        h_out, w_out = h_out // 2, w_out // 2
        
        conv3 = nn.Conv2d(32, 20, kernel_size=3, padding=1)
        h_out, w_out = out_dimensions(conv3, h_out, w_out)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        h_out, w_out = h_out // 2, w_out // 2
        
        self.conv_seq = nn.Sequential(conv1, pool1, nn.ReLU(), conv2, pool2, nn.ReLU(), conv3, pool3, nn.ReLU())
        
        self.fc1 = nn.Linear(20 * h_out * w_out, out_features=120)
        self.fc2 = nn.Linear(120, out_features=10)
        
        self.dimensions_final = (20, h_out, w_out)
        
    def forward(self, x):
        x = self.conv_seq(x)
        
        n_channels, h, w = self.dimensions_final
        x = x.view(-1, n_channels * h * w)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


Model(
  (conv_seq): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU()
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (5): ReLU()
    (6): Conv2d(32, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): ReLU()
  )
  (fc1): Linear(in_features=320, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=10, bias=True)
)
(20, 4, 4)
```

### Results

Best results so far:
- Epoch 1: Train loss: 1.848491627172408, Validation loss 1.5504446029663086
- Accuracy on the training set: 32.778 %, Accuracy on the validation set: 42.78 %
- Epoch 2: Train loss: 1.4202720697347124, Validation loss 1.3295255899429321
- Accuracy on the training set: 49.096 %, Accuracy on the validation set: 51.68 %
- Epoch 3: Train loss: 1.2372884644542226, Validation loss 1.1610671281814575
- Accuracy on the training set: 55.89 %, Accuracy on the validation set: 59.08 %
- Epoch 4: Train loss: 1.1121766908155064, Validation loss 1.0628730058670044
- Accuracy on the training set: 60.644 %, Accuracy on the validation set: 62.84 %
- Epoch 5: Train loss: 1.013532846308029, Validation loss 0.9866827130317688
- Accuracy on the training set: 64.416 %, Accuracy on the validation set: 65.22 %

**With accuracy on the test set: 65.5 % !!!**

**With batch_size=32 even better!!**

- Epoch 5: Train loss: 0.913563001521947, Validation loss 0.9622323513031006
- Accuracy on the training set: 68.066 %, Accuracy on the validation set: 66.98 %
- Accuracy on the **test set**: 66.36 %

### Pooling test on so far best Model

Only MaxPooling: **SMALL OVERFITTING**
- Epoch 1: Train loss: 1.7114025510737931, Validation loss 1.5297913551330566
- training set: 37.822 %, validation set: 43.78 %
- Epoch 2: Train loss: 1.2809194856474053, Validation loss 1.264981985092163
- training set: 54.344 %, validation set: 55.1 %
- Epoch 3: Train loss: 1.0790595119951325, Validation loss 1.1321457624435425
- training set: 61.854 %, validation set: 59.62 %
- Epoch 4: Train loss: 0.9500823815854329, Validation loss 1.0168273448944092
- training set: 66.708 %, validation set: 64.64 %
- Epoch 5: Train loss: 0.8597299287277997, Validation loss 1.0052814483642578
- training set: 69.736 %, validation set: 65.12 %
- **test** set: 66.26 %

Avg first:  **SMALL OVERFITTING**
- Epoch 1: Train loss: 1.7712662868292304, Validation loss 1.560899257659912
- training set: 35.688 %, validation set: 42.44 %
- Epoch 2: Train loss: 1.356797530875325, Validation loss 1.3674172163009644
- training set: 51.318 %, validation set: 51.3 %
- Epoch 3: Train loss: 1.1696416714293638, Validation loss 1.1770596504211426
- training set: 58.434 %, validation set: 58.16 %
- Epoch 4: Train loss: 1.0422535224824248, Validation loss 1.1236618757247925
- training set: 63.278 %, validation set: 60.52 %
- Epoch 5: Train loss: 0.9503637230403897, Validation loss 1.0693359375
- training set: 66.666 %, validation set: 63.58 %
- **test** set: 62.42 %
  
Avg in the middle: **Winner**
- Epoch 1: Train loss: 1.848491627172408, Validation loss 1.5504446029663086
- training set: 32.778 %, validation set: 42.78 %
- Epoch 2: Train loss: 1.4202720697347124, Validation loss 1.3295255899429321
- training set: 49.096 %, validation set: 51.68 %
- Epoch 3: Train loss: 1.2372884644542226, Validation loss 1.1610671281814575
- training set: 55.89 %, validation set: 59.08 %
- Epoch 4: Train loss: 1.1121766908155064, Validation loss 1.0628730058670044
- training set: 60.644 %, validation set: 62.84 %
- Epoch 5: Train loss: 1.013532846308029, Validation loss 0.9866827130317688
- training set: 64.416 %, validation set: 65.22 %
- **test** set: 66.36 %

Avg last: **OVERFITTING**
- Epoch 1: Train loss: 1.7580246846033682, Validation loss 1.575137734413147
- training set: 36.198 %, validation set: 42.76 %
- Epoch 2: Train loss: 1.3550818647929512, Validation loss 1.3738819360733032
- training set: 51.48 %, validation set: 51.22 %
- Epoch 3: Train loss: 1.1732356291662327, Validation loss 1.190821886062622
- training set: 58.122 %, validation set: 57.66 %
- Epoch 4: Train loss: 1.0398968402117548, Validation loss 1.0797159671783447
- training set: 63.102 %, validation set: 61.64 %
- Epoch 5: Train loss: 0.9355919743034219, Validation loss 1.1028292179107666
- training set: 66.89 %, validation set: 61.7 %
- **test** set: 61.62 %

Max - Avg- Avg: < 65%
- Epoch 1: Train loss: 1.7973564302623843, Validation loss 1.649321436882019
- training set: 34.588 %, validation set: 38.72 %
- Epoch 2: Train loss: 1.3881691775486702, Validation loss 1.5427002906799316
- training set: 50.274 %, validation set: 45.48 %
- Epoch 3: Train loss: 1.223839771457765, Validation loss 1.2038308382034302
- training set: 56.444 %, validation set: 57.08 %
- Epoch 4: Train loss: 1.1012258160899864, Validation loss 1.103435754776001
- training set: 60.896 %, validation set: 60.72 %
- Epoch 5: Train loss: 1.0072282843687408, Validation loss 1.0180456638336182
- training set: 64.43 %, validation set: 64.42 %
- **test** set: 64.22 %


## Assignment 3

- Download dataset from hugginface when using kaggle
- unknown token?
- Dataset: Pay attention to return type
- Model: Use general architecture, maybe bidirectional (works better)
- 