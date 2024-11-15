######################
# Import libraries
######################
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import loadtxt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher
import joblib
import pickle
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from IPython.display import HTML
from molvs import standardize_smiles
from math import pi
import zipfile
import base64
from pathlib import Path


######################
# Page Title
######################

st.write("<h1 style='text-align: center; color: #FF7F50;'> HDAC3_VS_assistant</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center; color: #483D8B;'> The application provides an alternative method for assessing the potential of chemicals to be Histone deacetylase 3 (HDAC3) inhibitors.</h3>", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)


with col1:
   st.header("Machine learning")
   st.image("figures/machine-learning.png", width=125)
   st.text_area('Text to analyze', '''This application makes predictions based on Quantitative Structure-Activity Relationship (QSAR) models build on curated datasets generated from scientific articles. The  models were developed using open-source chemical descriptors based on Morgan fingerprints and MACCS-166 keys, along with the gradient boosting method (GBM) and  support vector machine (SVM), using Python 3.10''', height=350, label_visibility="hidden" )


with col2:
   st.header("OECD principles")
   st.image("figures/target.png", width=125)
   st.text_area('Text to analyze', '''We follow the best practices for model development and validation recommended by guidelines of the Organization for Economic Cooperation and Development (OECD). The applicability domain (AD) of the models was calculated as Dcutoff = ⟨D⟩ + Zs, where «Z» is a similarity threshold parameter defined by a user (0.5 in this study) and «⟨D⟩» and «s» are the average and standard deviation, respectively, of all Euclidian distances in the multidimensional descriptor space between each compound and its nearest neighbors for all compounds in the training set. ''', height=350, label_visibility="hidden" )
# st.write('Sentiment:', run_sentiment_analysis(txt))


with col3:
   st.header("Acute toxicity")
   st.image("figures/mouse.png", width=125)
   st.text_area('Text to analyze', '''The application also allows to predict the level of toxicity (mouse, intravenous, LD50) of the studied compounds. One of the most common methods of administration of antitumor drugs is an intravenous one; it has a number of pharmacokinetic features, e.g., short-term but high active-substance concentration peaks, which can ultimately increase the acute toxicity of the drug. Therefore, a preliminary estimation of the toxicity of potential drugs during intravenous administration is extremely important ''', height=350, label_visibility="hidden" )
with col4:
   st.header("Structural Alerts")
   st.image("figures/alert.png", width=125)
   st.text_area('Text to analyze', '''Brenk filters which consists in a list of 105 fragments to be putatively toxic, chemically reactive, metabolically unstable or to bear properties responsible for poor pharmacokinetics. PAINS  are molecules containing substructures showing potent response in assays irrespective of the protein target. Such fragments, yielding false positive biological output.''', height=350, label_visibility="hidden" )

with open("manual.pdf", "rb") as file:
    btn=st.download_button(
    label="Click to download brief manual",
    data=file,
    file_name="manual of HDAC3_VS_assistant web application.pdf",
    mime="application/octet-stream"
)

def rdkit_numpy_convert(f_vs):
    output = []
    for f in f_vs:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
        return np.asarray(output)
def calcfp(mol,funcFPInfo=dict(radius=2, nBits=1024, useFeatures=False, useChirality=False)):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, **funcFPInfo)
    fp = pd.Series(np.asarray(fp))
    fp = fp.add_prefix('Bit_')
    return fp


st.write("<h3 style='text-align: center; color: black;'> Step 1. Draw molecule or select input molecular files.</h3>", unsafe_allow_html=True)
files_option1 = st.selectbox('', ('Draw the molecule and click the "Apply" button','SMILES', '*CSV file containing SMILES', 'MDL multiple SD file (*.sdf)'))
if files_option1 == 'Draw the molecule and click the "Apply" button':
    smiles = st_ketcher(height=400)
    st.write('''N.B. To start the step 2 (prediction), don't forget to click the "Apply" button''')
    st.write('If you want to create a new chemical structure, press the "Reset" button')
    st.write(f'The SMILES of the created  chemical: "{smiles}"')
    if len(smiles)!=0:
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles = False)
        smiles=standardize_smiles(canon_smi)
        m = Chem.MolFromSmiles(smiles)
        inchi = str(Chem.MolToInchi(m))
        
if files_option1 == 'SMILES':
    SMILES_input = ""
    compound_smiles = st.text_area("Enter only one structure as a SMILES", SMILES_input)
    if len(compound_smiles)!=0:
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(compound_smiles),isomericSmiles = False)
        smiles=standardize_smiles(canon_smi)
        m = Chem.MolFromSmiles(smiles)
        inchi = str(Chem.MolToInchi(m))
        im = Draw.MolToImage(m)
        st.image(im)

if files_option1 == '*CSV file containing SMILES':     
    # Read input
    uploaded_file = st.file_uploader('The file should contain only one column with the name "SMILES"')
    if uploaded_file is not None:
        df_ws=pd.read_csv(uploaded_file, sep=';')
        count=0
        failed_mols = []
        bad_index=[]
        index=0
        for i in df_ws.SMILES: 
            index+=1           
            try:
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(i),isomericSmiles = False)
                df_ws.SMILES = df_ws.SMILES.replace (i, canon_smi)             
            except:
                failed_mols.append(i)
                bad_index.append(index)
                canon_smi='wrong_smiles'
                count+=1
                df_ws.SMILES = df_ws.SMILES.replace (i, canon_smi)
        st.write('CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:')
        st.write(f'Original data: {len(df_ws)} molecules')
        st.write(f'Failed data: {count} molecules')

        if len(failed_mols)!=0:
            number =[]
            for i in range(len(failed_mols)):
                number.append(str(i+1))
            
            
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': bad_index, 'SMILES of wrong structure: ': failed_mols, 'No.': number}, index=None)
            bad_molecules = bad_molecules.set_index('No.')
            st.dataframe(bad_molecules)


        moldf = []
        for i,record in enumerate(df_ws.SMILES):
            if record!='wrong_smiles':
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(record),isomericSmiles = False)
                standard_record = standardize_smiles(canon_smi)
                m = Chem.MolFromSmiles(standard_record)
                moldf.append(m)
        
        st.write('Kept data: ', len(moldf), 'molecules') 

# Read SDF file 
if files_option1 == 'MDL multiple SD file (*.sdf)':
    uploaded_file = st.file_uploader("Choose a SDF file")
    if uploaded_file is not None:
        st.header('CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:')
        supplier = Chem.ForwardSDMolSupplier(uploaded_file,sanitize=False)
        failed_mols = []
        all_mols =[]
        wrong_structure=[]
        wrong_smiles=[]
        bad_index=[]
        for i, m in enumerate(supplier):
            structure = Chem.Mol(m)
            all_mols.append(structure)
            try:
                Chem.SanitizeMol(structure)
            except:
                failed_mols.append(m)
                wrong_smiles.append(Chem.MolToSmiles(m))
                wrong_structure.append(str(i+1))
                bad_index.append(i)

        
        st.write('Original data: ', len(all_mols), 'molecules')
        st.write('Failed data: ', len(failed_mols), 'molecules')
        if len(failed_mols)!=0:
            number =[]
            for i in range(len(failed_mols)):
                number.append(str(i+1))
            
            
            bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'No.': number}, index=None)
            bad_molecules = bad_molecules.set_index('No.')
            st.dataframe(bad_molecules)

        # Standardization SDF file
        all_mols[:] = [x for i,x in enumerate(all_mols) if i not in bad_index] 
        records = []
        for i in range(len(all_mols)):
            record = Chem.MolToSmiles(all_mols[i])
            records.append(record)
        
        moldf = []
        for i,record in enumerate(records):
            canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(record),isomericSmiles = False)
            standard_record = standardize_smiles(canon_smi)
            m = Chem.MolFromSmiles(standard_record)
            moldf.append(m)
        
        st.write('Kept data: ', len(moldf), 'molecules') 


class Models():
    def __init__(self,activity:str, way_exp_data:str, way_model:str, descripror_way_zip, descriprors:list,  path_feature_name_rfecv:str, model_AD_limit:float):
        self.activity=activity
        self.way_exp_data=way_exp_data
        self.way_model=way_model
        self.descripror_way_zip=descripror_way_zip
        self.descriprors=descriprors       
        self.path_feature_name_rfecv=path_feature_name_rfecv
        self.model_AD_limit=model_AD_limit
        # Load model and experimental dates
        self.model = pickle.load(open(self.way_model, 'rb'))
        self.df_exp = pd.read_csv(self.way_exp_data)
        self.res = (self.df_exp.groupby("inchi").apply(lambda x: x.drop(columns="inchi").to_dict("records")).to_dict())

        # Calculate molecular descriptors
        if self.activity=='HDAC3':
            self.desc_ws = calcfp(m)
            self.path = Path(self.path_feature_name_rfecv)
            self.feature_name_rfecv_MF = self.path.read_text().splitlines()
            self.f_vs_red=self.desc_ws[self.feature_name_rfecv_MF]
            self.X = self.f_vs_red.to_numpy().reshape(1, -1)
        else:
            self.f_vs=[MACCSkeys.GenMACCSKeys(m)]
            self.X = rdkit_numpy_convert(self.f_vs)

        self.zf = zipfile.ZipFile(self.descripror_way_zip) 
        self.df = pd.read_csv(self.zf.open(self.descriprors))
        self.x_tr=self.df.to_numpy()

class one_molecules(Models):                
    def seach_predic(self):
        # search experimental activity value
        if inchi in self.res:
            if self.activity=='HDAC3':
                exp=round(self.res[inchi][0]['pchembl_value_mean'],2)
                std=round(self.res[inchi][0]['pchembl_value_std'],4)
                chembl_id=str(self.res[inchi][0]['molecule_chembl_id']) 
                value_pred_tox='see experimental value'
                cpd_AD_vs_tox='-' 
            else:
                exp_tox=float(self.res[inchi][0]['TOX_VALUE'])
                cas_id=str(self.res[inchi][0]['CAS_Number'])
                value_pred_tox='see experimental value'
                cpd_AD_vs_tox='-'            
        else:
         #Predict activity
            if self.activity=='HDAC3':
                y_pred_con_act = self.model.predict(self.X)           
                value_pred_tox=round(y_pred_con_act[0], 3)            
                # Estimination AD for activity
                neighbors_k_vs_tox = pairwise_distances(self.x_tr, Y=self.X, n_jobs=-1)
                neighbors_k_vs_tox.sort(0)
                similarity_vs_tox = neighbors_k_vs_tox
                cpd_value_vs_tox = similarity_vs_tox[0, :]
                cpd_AD_vs_tox = np.where(cpd_value_vs_tox <= self.model_AD_limit, "Inside AD", "Outside AD")
                exp="-"
                std='-'
                chembl_id="not detected"
            else:
                y_pred_con_tox = self.model.predict(self.X)           
                y_pred_con_tox_t=y_pred_con_tox[0]
                MolWt=ExactMolWt(Chem.MolFromSmiles(smiles))
                value_pred_tox=round((10**(y_pred_con_tox_t*-1)*1000)*MolWt, 4)
                # Estimination AD for toxicity
                neighbors_k_vs_tox = pairwise_distances(self.x_tr, Y=self.X, n_jobs=-1)
                neighbors_k_vs_tox.sort(0)
                similarity_vs_tox = neighbors_k_vs_tox
                cpd_value_vs_tox = similarity_vs_tox[0, :]
                cpd_AD_vs_tox = np.where(cpd_value_vs_tox <= self.model_AD_limit, "Inside AD", "Outside AD")
                exp_tox="-"
                cas_id="not detected"


        if self.activity=='Toxicity':
            st.header('**Prediction results:**')    
            common_inf = pd.DataFrame({'SMILES':smiles, 'Predicted value toxicity, mice, intravenous, Ld50, mg/kg': value_pred_tox,
                'Applicability domain_tox': cpd_AD_vs_tox,
                'Experimental value toxicity, mice, intravenous, Ld50, mg/kg': exp_tox, 
                'CAS number': cas_id}, index=[1])
            predictions_pred=common_inf.astype(str) 
            st.dataframe(predictions_pred)

        else:
            st.header('**Prediction results:**')             
            common_inf = pd.DataFrame({'SMILES':smiles, 'Predicted value pIC50': value_pred_tox,
            'Applicability domain_HDAC3': cpd_AD_vs_tox,
            'Experimental value value pIC50': exp, 'STD': std, 
            'chembl_ID': chembl_id}, index=[1])
            predictions_pred=common_inf.astype(str) 
            st.dataframe(predictions_pred) 

class set_molecules(Models):    
    def seach_predic_csv(self):        
        if self.activity=='HDAC3':
            # search experimental value     
            exp_tox=[]
            std=[]
            chembl_id=[]
            y_pred_con_tox=[]
            cpd_AD_vs_tox=[]
            struct=[]
            number =[]
            count=0
            for m in moldf:
                inchi = str(Chem.MolToInchi(m))
                i=Chem.MolToSmiles(m)
                struct.append(i)
                # search experimental toxicity value
                if inchi in self.res:
                    exp_tox.append(self.res[inchi][0]['pchembl_value_mean'])
                    std.append(self.res[inchi][0]['pchembl_value_std'])
                    chembl_id.append(str(self.res[inchi][0]['molecule_chembl_id']))
                    y_pred_con_tox.append('see experimental value')
                    cpd_AD_vs_tox.append('-')
                    count+=1         
                    number.append(count)
                    
                else:
                    # Estimination AD for toxicity
                    neighbors_k_vs_tox = pairwise_distances(self.x_tr, Y=self.X, n_jobs=-1)
                    neighbors_k_vs_tox.sort(0)
                    similarity_vs_tox = neighbors_k_vs_tox
                    cpd_value_vs_tox = similarity_vs_tox[0, :]
                    cpd_AD_vs_tox_r = np.where(cpd_value_vs_tox <= self.model_AD_limit, "Inside AD", "Outside AD")
                    # calculate toxicity
                    y_pred_tox = self.model.predict(self.X)                
                    value_ped_tox=round(y_pred_tox[0], 4)
                    y_pred_con_tox.append(value_ped_tox)
                    cpd_AD_vs_tox.append(cpd_AD_vs_tox_r[0])
                    exp_tox.append("-")
                    chembl_id.append("not detected")
                    count+=1         
                    number.append(count)
        else:
            # search experimental value     
            exp_tox=[]
            cas_id=[]
            y_pred_con_tox=[]
            cpd_AD_vs_tox=[]
            struct=[]
            number =[]
            count=0
            for m in moldf:
                inchi = str(Chem.MolToInchi(m))
                i=Chem.MolToSmiles(m)
                struct.append(i)
                # search experimental toxicity value
                if inchi in self.res:
                    exp_tox.append(self.res[inchi][0]['TOX_VALUE'])
                    cas_id.append(str(self.res[inchi][0]['CAS_Number']))
                    y_pred_con_tox.append('see experimental value')
                    cpd_AD_vs_tox.append('-')
                    count+=1         
                    number.append(count)
                    
                else:
                    # Estimination AD for toxicity
                    neighbors_k_vs_tox = pairwise_distances(self.x_tr, Y=self.X, n_jobs=-1)
                    neighbors_k_vs_tox.sort(0)
                    similarity_vs_tox = neighbors_k_vs_tox
                    cpd_value_vs_tox = similarity_vs_tox[0, :]
                    cpd_AD_vs_tox_r = np.where(cpd_value_vs_tox <= self.model_AD_limit, "Inside AD", "Outside AD")
                    # calculate toxicity
                    y_pred_tox = self.model.predict(self.X)                    
                    MolWt=ExactMolWt(m)
                    value_ped_tox=(10**(y_pred_tox*-1)*1000)*MolWt
                    value_ped_tox=round(value_ped_tox[0], 4)
                    y_pred_con_tox.append(value_ped_tox)
                    cpd_AD_vs_tox.append(cpd_AD_vs_tox_r[0])
                    exp_tox.append("-")
                    cas_id.append("not detected")
                    count+=1         
                    number.append(count) 


        # visualization of the results
        if self.activity=='Toxicity':
            common_inf = pd.DataFrame({'SMILES':struct, 'No.': number,'Predicted value toxicity, rat, oral, Ld50, mg/kg': y_pred_con_tox,
            'Applicability domain_tox': cpd_AD_vs_tox,
            'Experimental value toxicity, Ld50': exp_tox,
            'CAS number': cas_id}, index=None)
            predictions_pred = common_inf.set_index('No.')
            predictions_pred=predictions_pred.astype(str)
            st.dataframe(predictions_pred)

        else:
            common_inf = pd.DataFrame({'SMILES':struct, 'No.': number,'Predicted value pIC50': y_pred_con_tox,
            'Applicability domain_tox': cpd_AD_vs_tox,
            'Experimental value toxicity, Ld50': exp_tox,
            'CAS number': chembl_id}, index=None)
            predictions_pred = common_inf.set_index('No.')
            predictions_pred=predictions_pred.astype(str)
            st.dataframe(predictions_pred)

        def convert_df(df):
            return df.to_csv().encode('utf-8')  
        csv = convert_df(predictions_pred)

        st.download_button(
            label="Download results of prediction as CSV",
            data=csv,
            file_name='Results.csv',
            mime='text/csv',
        )

class Med_chem_one():
    def __init__(self, propetis:str, way_exp_data:list):
        self.propetis=propetis
        self.way_exp_data=way_exp_data
        self.substructures_df = pd.read_csv(self.way_exp_data, sep="\s+")
        # Converting SMARTS substructures into RDKit molecules
        self.substructure_mols = [(row['name'], Chem.MolFromSmarts(row['smarts'])) for _, row in self.substructures_df.iterrows()]
        if self.propetis=='vip_subst':
            # Creating a topological fingerprint for the original molecule
            self.mol_fp = FingerprintMols.FingerprintMol(m)
        # A dictionary for found substructures with their atomic indexes
        self.found_substructures = {}
        for name, substructure in self.substructure_mols:
            if substructure:
                match = m.GetSubstructMatch(substructure)
                if match:
                    self.found_substructures[name] = match
        # Checking if substructures are found
        if self.found_substructures:
            # A passage through each found substructure and a display of a molecule with isolated atoms
                for name, atoms in self.found_substructures.items():
                    st.write(f"The found {self.propetis}: {name}")
                    # Calculating the Tanimoto coefficient
                    self.substructure_mol = Chem.MolFromSmarts(self.substructures_df[self.substructures_df['name'] == name]['smarts'].values[0])
                    if self.propetis=='vip_subst':
                        self.sub_fp = FingerprintMols.FingerprintMol(self.substructure_mol)
                        self.tanimoto_similarity = DataStructs.TanimotoSimilarity(self.mol_fp, self.sub_fp)
                        st.write(f"Tanimoto coefficient: {self.tanimoto_similarity:.2f}")
                    if self.propetis=='Brenk_SA':
                        st.header('The Structural Alerts or Brenk filters [DOI:10.1002/cmdc.200700139] contain substructures with undesirable pharmacokinetics or toxicity*')
                    if self.propetis=='Pains':
                        st.header('*Filter for PAINS*')
                    # visualization of a molecule with a highlighted sub-structure
                    img = Draw.MolToImage(m, highlightAtoms=atoms, size=(300, 300))
                    st.image(img)  # Display an image
        else:
            st.write(f"The {self.propetis} are not found in the molecule.")

st.write("<h3 style='text-align: center; color: black;'> Step 2. Select prediction of HDAC3 inhibitor activity or acute toxicity to mice or substructural search for preferred or undesirable fragments</h3>", unsafe_allow_html=True)
files_option2 = st.selectbox('', ('HDAC3','Toxicity', 'Substructural search, PAINS, Brenk structural alerts'))
if (files_option1 =='Draw the molecule and click the "Apply" button' or files_option1 =='SMILES')  and files_option2 =='HDAC3':
    if st.button('Run predictions!'):
        HDAC3_one=one_molecules('HDAC3', 'datasets/HDAC3_exp_data_inchi.csv', 'Models/HDAC3_GBR_MF_final_FS.pkl', 'Models/descriptorMF.zip',
                             'descriptorMF.csv', 'Models/feature_name_rfecv_MF.txt', 2.78)
        HDAC3_one.seach_predic()

if (files_option1 =='Draw the molecule and click the "Apply" button' or files_option1 =='SMILES')  and files_option2 == 'Toxicity':
    if st.button('Run predictions!'):
        Toxicity_one=one_molecules('Toxicity', 'datasets/mouse_intravenous_LD50_inchi.csv', 'Models/LD50_mouse_introvenus_SVM_MACCS.pkl',
                              'Models/x_tr_MACCS.zip', 'x_tr_MACCS.csv', 'Models/feature_name_rfecv_toxicity.txt',  2.45)
        Toxicity_one.seach_predic()
 
if (files_option1  =='*CSV file containing SMILES' or files_option1=='MDL multiple SD file (*.sdf)')  and files_option2 =='HDAC3':
    if st.button('Run predictions!'):
        HDAC3_set=set_molecules('HDAC3', 'datasets/HDAC3_exp_data_inchi.csv', 'Models/HDAC3_GBR_MF_final_FS.pkl', 'Models/descriptorMF.zip',
                             'descriptorMF.csv', 'Models/feature_name_rfecv_MF.txt', 2.78)
        HDAC3_set.seach_predic_csv()

if (files_option1  =='*CSV file containing SMILES' or files_option1=='MDL multiple SD file (*.sdf)')  and files_option2 =='Toxicity':
    if st.button('Run predictions!'):
        Toxicity_set=set_molecules('Toxicity', 'datasets/mouse_intravenous_LD50_inchi.csv', 'Models/LD50_mouse_introvenus_SVM_MACCS.pkl',
                              'Models/x_tr_MACCS.zip', 'x_tr_MACCS.csv', 'Models/feature_name_rfecv_toxicity.txt',  2.45)
        Toxicity_set.seach_predic_csv()
if (files_option1 =='Draw the molecule and click the "Apply" button' or files_option1 =='SMILES')  and files_option2 =='Substructural search, PAINS, Brenk structural alerts':
    if st.button('Run predictions!'):
        Substructural_search_one=Med_chem_one('fragments that increase the activity to inhibit HDAC3', 'datasets/vip_substructures.csv')
        Brenk_SA=Med_chem_one('Brenk filter', 'datasets/unwanted_substructures.csv')
        Pains=Med_chem_one('PAINS', 'datasets/PAINS.csv')
        