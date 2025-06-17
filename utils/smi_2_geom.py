#!/usr/bin/env python 
from rdkit import Chem
from rdkit.Chem import AllChem


def smi_2_geom(smi:str) -> tuple:
    """
    Convert a SMILES string to a 3D geometry using RDKit.
    
    Parameters:
    smi (str): The SMILES string to convert.
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object with 3D coordinates.
    """
    # Create a molecule from the SMILES string
    try:
        mol = Chem.MolFromSmiles(smi)
    
        # Add hydrogens to the molecule
        mol = Chem.AddHs(mol)
    
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol,maxIters=100)
        coord = []; symbol = []
        conformer = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            coord.append([pos.x, pos.y, pos.z])
            symbol.append(mol.GetAtomWithIdx(i).GetSymbol())
        return coord, symbol, smi
    except Exception as e:
        return None
