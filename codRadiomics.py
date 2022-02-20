from radiomics import featureextractor
import SimpleITK as sitk
import six
from numpy import savetxt
import glob
import cv2
import numpy as np
import mahotas as mt


dataCompleto= []
features = [];

imagensMild = []
imagensSaudaveis = []
imagensModerate = []
imagensProliferate = []
imagensSevere = []



CV2imagensMild = []
CV2imagensSaudaveis = []
CV2imagensModerate = []
CV2imagensProliferate = []
CV2imagensSevere = []



maskMild = []
maskSaudavel = []
maskModerate = []
maskProliferate = []
maskSevere = []


#mask = sitk.ReadImage('./3.0/dataBase/mask.jpeg')
name = [];


params = { 'force2D': True,
          'force2Dextration': True,
          'force2Ddimension': True}


extractor = featureextractor.RadiomicsFeatureExtractor(**params)


#----------------------------------------------Listar Imagens de cada categoria ------------------------------------------------


def lerImagensSaudaveis():
    for f in glob.glob('./database_Processada/imgSaudavel/*.png'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        imagensSaudaveis.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens
        CV2imagensSaudaveis.append(cv2.imread(f,0))


def lerImagensMild():     
    for f in glob.glob('./database_Processada/imgMild/*.png'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        imagensMild.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens
        CV2imagensMild.append(cv2.imread(f,0))


def lerImagensModerate():
    for f in glob.glob('./database_Processada/imgModerate/*.png'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        imagensModerate.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens
        CV2imagensModerate.append(cv2.imread(f,0))        
        
        
def lerImagensProliferate():
    for f in glob.glob('./database_Processada/imgProliferate/*.png'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens      
        imagensProliferate.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens     
        CV2imagensProliferate.append(cv2.imread(f,0))        
        
        
def lerImagensSevere():
    for f in glob.glob('./database_Processada/imgSevere/*.png'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        imagensSevere.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens
        CV2imagensSevere.append(cv2.imread(f,0))

#--------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------Listar Mascaras -----------------------------------------------------------


def lerMascarasSaudaveis():
      for f in glob.glob('./database_mask/imgSaudavel/*.jpeg'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        maskSaudavel.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens


def lerMascarasMild():
      for f in glob.glob('./database_mask/imgMild/*.jpeg'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        maskMild.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens
        

def lerMascarasModerate():
      for f in glob.glob('./database_mask/imgModerate/*.jpeg'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        maskModerate.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens


def lerMascarasProliferate():
      for f in glob.glob('./database_mask/imgProliferate/*.jpeg'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        maskProliferate.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens


def lerMascarasSevere():
      for f in glob.glob('./database_mask/imgSevere/*.jpeg'): #lista todos os arquivos da pasta especifica
        print(f)    #conferindo se pegou todas as imagens    
        maskSevere.append(sitk.ReadImage(f)) #lendo as imagens da pasta para o array Imagens

#---------------------------------------------------------------------------------------------------------------------------



def extract_features_Saudaveis(imagensSaudaveis):
    print('....Features_Saudaveis....')
    features = []
    for i in range(len(imagensSaudaveis)):
        #radiomics = extract_radiomics(imagensSaudaveis[i], maskSaudavel[i])
        radiomics = extract_radiomics(imagensSaudaveis[i], maskSaudavel[i])
        
        image = CV2imagensSaudaveis[i]
        
        lbp = mt.features.lbp(image, radius=8, points=6)      
        haralick = mt.features.haralick(image)
        zernike = mt.features.zernike(image,10,10)    
        tas = mt.features.tas(image)

        
        features = np.concatenate((radiomics,lbp,tas,haralick,zernike),axis=None)       
        features = np.append(features, 0)
        
        #print(features)
        print(len(features))
        dataCompleto.append(features)


def extract_features_Mild(imagensMild):
    print('....Features_Mild....')
    features = []
    for i in range(len(imagensMild)):
        #radiomics = extract_radiomics(imagensDoentes[i], maskEarly[i])
        radiomics = extract_radiomics(imagensMild[i], maskMild[i])
        
        image = CV2imagensMild[i]
        
        lbp = mt.features.lbp(image, radius=8, points=6)  
        haralick = mt.features.haralick(image)
        zernike = mt.features.zernike(image,10,10)      
        tas = mt.features.tas(image)      
        
        features = np.concatenate((radiomics,lbp,tas,haralick,zernike),axis=None)       
        features = np.append(features, 1)
        
        print(features)
        dataCompleto.append(features)
        
        
def extract_features_Moderate(imagensModerate):
    print('....Features_Moderate....')
    features = []
    for i in range(len(imagensModerate)):
        #radiomics = extract_radiomics(imagensDoentes[i], maskEarly[i])
        radiomics = extract_radiomics(imagensModerate[i], maskModerate[i])
       
        image = CV2imagensModerate[i]
        
        lbp = mt.features.lbp(image, radius=8, points=6)       
        haralick = mt.features.haralick(image)
        zernike = mt.features.zernike(image,10,10)    
        tas = mt.features.tas(image)

        
        features = np.concatenate((radiomics,lbp,tas,haralick,zernike),axis=None)       
        features = np.append(features, 1)
        
        print(features)
        dataCompleto.append(features)
        
        
def extract_features_Proliferate(imagensProliferate):
    print('....Features_Proliferate....')
    features = []
    for i in range(len(imagensProliferate)):
        #radiomics = extract_radiomics(imagensDoentes[i], maskEarly[i])
        radiomics = extract_radiomics(imagensProliferate[i], maskProliferate[i])
        
        image = CV2imagensProliferate[i]
        
        lbp = mt.features.lbp(image, radius=8, points=6)  
        haralick = mt.features.haralick(image)
        zernike = mt.features.zernike(image,10,10)      
        tas = mt.features.tas(image)      
        
        features = np.concatenate((radiomics,lbp,tas,haralick,zernike),axis=None)       
        features = np.append(features, 1)
        
        print(features)
        dataCompleto.append(features)
        
    
def extract_features_Severe(imagensSevere):
    print('....Features_Severe....')
    features = []
    for i in range(len(imagensSevere)):
        #radiomics = extract_radiomics(imagensDoentes[i], maskEarly[i])
        radiomics = extract_radiomics(imagensSevere[i], maskSevere[i])
       
        image = CV2imagensSevere[i]
        
        lbp = mt.features.lbp(image, radius=8, points=6)  
        haralick = mt.features.haralick(image)
        zernike = mt.features.zernike(image,10,10)      
        tas = mt.features.tas(image)      
        
        features = np.concatenate((radiomics,lbp,tas,haralick,zernike),axis=None)       
        features = np.append(features, 1)
        
        print(features)
        dataCompleto.append(features)
        


def extract_radiomics(imagem,mascara):
    data = []
    color_channel = 0
    #im = sitk.ReadImage('./teste.jpeg')
    im = imagem
    selector = sitk.VectorIndexSelectionCastImageFilter()
    selector.SetIndex(color_channel)
    im = selector.Execute(im)
    #results = extractor.execute(im, './mteste.jpg', 1)
    results = extractor.execute(im, mascara, 1)
    i =0
    for key, val in six.iteritems(results):
        if isinstance(val, sitk.Image):  # Feature map
            sitk.WriteImage(val, key + '.jpeg', True)
            print("Stored feature %s in %s" % (key, key + ".jpeg"))
        else:  # Diagnostic information
            print("\t%s: %s" %(key, val))
            #print(val)
            name.append(key)
        if(i>21):
            data.append(val)
        i = i + 1; 
    print(len(data))
    return data;



lerImagensSaudaveis()
lerMascarasSaudaveis()

lerImagensMild()
lerMascarasMild()

lerImagensModerate()
lerMascarasModerate()

lerImagensProliferate()
lerMascarasProliferate()

lerImagensSevere()
lerMascarasSevere()



extract_features_Saudaveis(imagensSaudaveis)
extract_features_Mild(imagensMild)
extract_features_Moderate(imagensModerate)
extract_features_Proliferate(imagensProliferate)
extract_features_Severe(imagensSevere)


savetxt("./dataset.csv", dataCompleto, delimiter=',')

































