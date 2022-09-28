from .__importList__ import *
from .utilities import *
from .features import *
from .dataDefinitions import *

def loadFemData(path, noiseLevel = 0., noiseType = 'displacement', denoisedDisplacements = None):
    """
    Load finite element data and add noise (optional).
    Note that the loaded finite element data might already be perturbed by noise.
    In that case, adding additional noise is not necessary.

    _Input Arguments_

    - `path` - path to finite element data

    - `AD` - specify if automatic differention is needed

    - `noiseLevel` - noise level

    - `noiseType` - specify whether noise should be added to displacements or strains

    - `denoisedDisplacements` - pass denoised displacement data if available

    _Output Arguments_

    - `dataset` - finite element dataset

    ---

    """
    print('\n-----------------------------------------------------')
    print('Loading data: ', path)

    if(path[-1]=='/'):
        path=path[0:-1]

    numNodesPerElement = 3

    #nodal data
    df = pd.read_csv(path+'/output_nodes.csv',dtype=np.float64)

    if 'ux_orig' in df.columns and 'uy_orig' in df.columns:
        df.ux[df.bcx!=0] = df.ux_orig[df.bcx!=0]
        df.uy[df.bcy!=0] = df.uy_orig[df.bcy!=0]

    numNodes = df.shape[0]
    x_nodes = torch.tensor(df[['x','y']].values)

    u_nodes = torch.tensor(df[['ux','uy']].values)

    if(denoisedDisplacements is not None):
        u_nodes = denoisedDisplacements

    bcs_nodes = torch.tensor(df[['bcx','bcy']].round().astype(int).values)
    #convert bcs_nodes to booleans
    dirichlet_nodes = (bcs_nodes!=0)

    #noise
    if((noiseType=='displacement') or (noiseType=='strain')):
        pass
    else:
        raise ValueError('Incorrect noiseType argument!')


    noise_nodes = noiseLevel * torch.randn_like(u_nodes)
    noise_nodes[dirichlet_nodes] = 0.
    if(noiseType == 'displacement'):
        u_nodes += noise_nodes
        print('Applying noise to displacements:',noiseLevel)

    #reaction forces data
    numReactions = torch.max(bcs_nodes).item()
    df = pd.read_csv(path+'/output_reactions.csv',dtype=np.float64)
    reactions = []
    for i in range(numReactions):
        reactions.append(Reaction(bcs_nodes == (i+1),df['forces'][i]))


	#element data
    df = pd.read_csv(path+'/output_elements.csv',dtype=np.float64)
    numElements = df.shape[0]
    connectivity = []
    for i in range(numNodesPerElement):
        connectivity.append(torch.tensor(df['node'+str(i+1)].round().astype(int).tolist()))


	#integrator/shape-function data
    df = pd.read_csv(path+'/output_integrator.csv',dtype=np.float64)
    gradNa = []
    for i in range(numNodesPerElement):
        gradNa.append(torch.tensor(df[['gradNa_node'+str(i+1)+'_x','gradNa_node'+str(i+1)+'_y']].values))
    qpWeights = torch.tensor(df['qpWeight'].values)


    #element-wise rearrangement of displacements
    u = []
    for i in range(numNodesPerElement):
        u.append(u_nodes[connectivity[i],:])


    #computing deformation gradient at quadrature points
    dim=2
    voigtMap = [[0,1],[2,3]]

    F=torch.zeros(numElements,4)
    for a in range(numNodesPerElement):
        for i in range(dim):
            for j in range(dim):
                F[:,voigtMap[i][j]] += u[a][:,i] * gradNa[a][:,j]
    F[:,0] += 1.0
    F[:,3] += 1.0

    if(noiseType == 'strain'):
        F += noiseLevel * torch.randn_like(F)
        print('Applying noise to strains:',noiseLevel)

    #computing detF
    J = computeJacobian(F)

    #computing Cauchy-Green strain: C = F^T F
    C = computeCauchyGreenStrain(F)

    #computing strain invariants
    I1, I2, I3 = computeStrainInvariants(C)

    #activate gradients
    I1.requires_grad = True
    I2.requires_grad = True
    I3.requires_grad = True

    #computing invariant derivaties
    dI1dF = computeStrainInvariantDerivatives(F,1)
    dI2dF = computeStrainInvariantDerivatives(F,2)
    dI3dF = computeStrainInvariantDerivatives(F,3)

    #computing extended set of nonlinear features
    featureSet = FeatureSet()
    featureSet.features = computeFeatures(I1, I2, I3).detach()

    dataset = FemDataset(path,
        x_nodes, u_nodes, dirichlet_nodes,
        reactions,
        connectivity, gradNa, qpWeights,
        F, J, C,
        I1, I2, I3,
        dI1dF, dI2dF, dI3dF,
        featureSet)

    print('-----------------------------------------------------\n')

    return dataset
