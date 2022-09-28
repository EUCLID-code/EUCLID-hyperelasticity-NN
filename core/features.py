from .__importList__ import *

def computeFeatures(I1, I2, I3):
    """
    Compute the features dependent on the right Cauchy-Green strain invariants.
    Note that the features only depend on I1 and I3.

    _Input Arguments_

    - `I1` - 1st invariant

    - `I2` - 2nd invariant

    - `I3` - 3rd invariant

    _Output Arguments_

    - `x` - features

    ---

    """
    # numFeatures = 3
    # x = torch.zeros(I1.shape[0],numFeatures)

    K1 = I1 * torch.pow(I3,-1/3) - 3.0
    K2 = (I1 + I3 - 1) * torch.pow(I3,-2/3) - 3.0
    J = torch.sqrt(I3)
    K3 = (J-1)**2

    x = torch.cat((K1,K2,K3),1).float()
    return x

def getNumberOfFeatures():
    """
    Compute number of features.

    _Input Arguments_

    - _none_

    _Output Arguments_

    - `features.shape[1]` - number of features

    ---

    """
    features = computeFeatures(torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1))
    return features.shape[1]
