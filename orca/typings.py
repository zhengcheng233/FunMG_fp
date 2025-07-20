from pydantic import BaseModel,Field
from typing import Literal
from pathlib import Path
import json
from loguru import logger
import fchic 
from ase.data import chemical_symbols, atomic_numbers
import numpy as np
import subprocess

JobType = Literal['s0','s1','t1','absorption_spec','emission_spec']

path_prefix=Path('/vepfs/fs_users/chensq/project/funmg/runtime_data/tasks/dft')

class Params(BaseModel):
    nproc: int = 32
    mem: str = '64GB'
    method: str = 'b3lyp'
    basis: str  = 'def2svp'
    EmpiricalDispersion:str ='GD3BJ'
    td: int = -1 # f" td=(nstates={self.td})"
    opt:bool = False
    freq:bool = False
    charge:int = 0
    multiplicity: int = 1
    title: JobType|str = 's0'
    molecule_filename:str = 'input.xyz'

    @classmethod
    def parse_from_json(cls, json_path:Path):
        with open(json_path, 'r') as file:
            data = json.load(file)
    
        params=cls(**data)
        return params

    def to_str_lines(self) -> str:
        route = f"#P {self.method}/{self.basis}"
        if self.EmpiricalDispersion:
            route += f" EmpiricalDispersion={self.EmpiricalDispersion}"
        if self.opt:
            route += " opt"
        if self.freq:
            route += " freq"
        if self.td != -1:
            route += f" td=(nstates={self.td})"

        header_lines = [
            f"%nproc={self.nproc}",
            f"%mem={self.mem}",
            route,
            '',
            f'Job Type: {self.title}',
            '',
            f"{self.charge} {self.multiplicity}"
        ]

        header = "\n".join(header_lines) + "\n"
        return header


class Atom(BaseModel):
    symbol:str
    x:float
    y:float
    z:float

class Structure(BaseModel):
    atoms:list[Atom] = Field(default_factory=list)

    @classmethod
    def parse_from_xyz_file(cls,xyz_path:Path) -> "Structure":
        with open(xyz_path, 'r') as f:
            lines = f.readlines()

        total_atoms_nums = int(lines[0].strip())
        atoms = []
        for line in lines[2:]:
            sub_strs = line.strip().split()
            if len(sub_strs) == 4:
                atoms.append(
                    Atom(
                        symbol=sub_strs[0],
                        x=float(sub_strs[1]), 
                        y=float(sub_strs[2]), 
                        z=float(sub_strs[3])
                    )
                )

        if len(atoms) != total_atoms_nums:
            logger.warning(f'parse xyz to Structure warning: total_atoms_num-{total_atoms_nums} not equal parse_len-{len(atoms)}; {xyz_path.as_posix()}')
        
        return cls(atoms=atoms)
    
    @classmethod
    def  parse_from_fchk_file(cls,fchk_path:Path) -> "Structure":
        with open(fchk_path, 'r') as f:
            coord_s0_opt = fchic.deck_load(f, "Current cartesian coordinates")
            species_s0_opt = fchic.deck_load(f, "Atomic numbers")

        symbol_s0_opt = [chemical_symbols[i] for i in species_s0_opt] # type: ignore
        coord_s0_opt = np.array(coord_s0_opt)
        ## 注意采用fchk信息时候，单位是au
        coord_s0_opt = coord_s0_opt.reshape((-1,3)) * 0.5291772

        atoms = []
        for index,symbol in enumerate(symbol_s0_opt):
            atoms.append(Atom(
                symbol=symbol,
                x=coord_s0_opt[index][0],
                y=coord_s0_opt[index][1],
                z=coord_s0_opt[index][2],
            ))

        return cls(atoms=atoms)

    def to_str_lines(self) -> str:
        atom_str_list = [
            f"{atom.symbol:<2}  {atom.x:12.6f}  {atom.y:12.6f}  {atom.z:12.6f}"
            for atom in self.atoms
        ]
        structure_str = '\n'.join(atom_str_list) + "\n"
        return structure_str


class G16Input(BaseModel):
    params: Params
    structure: Structure

    def to_content(self)->str:
        key_word_str=self.params.to_str_lines()
        structure_str = self.structure.to_str_lines()
        return key_word_str + '\n' + structure_str + "\n"
    
    def to_file(self,path_prefix:Path):
        content = self.to_content()
        file_path = Path(path_prefix,f'{self.params.title}.com')
        with open(file_path,'w') as f:
            f.write(content)

    @classmethod
    def gen_input_file_and_return_self(cls, params:Params, path_prefix:Path=path_prefix, fchk_name:str='') -> "G16Input":
    
        if fchk_name:
            fchk_path = Path(path_prefix, fchk_name)
            structure=Structure.parse_from_fchk_file(fchk_path)
        else:
            molecule_path = Path(path_prefix, params.molecule_filename)
            structure=Structure.parse_from_xyz_file(molecule_path)

        g16_input = cls( params=params, structure=structure )
        g16_input.to_file(path_prefix)
        return g16_input
    
    def run(self):
        result = subprocess.run(['g16', f'{self.params.title}.com'],capture_output=True, text=True)
        result = subprocess.run(['formchk', f'{self.params.title}.chk', f'{self.params.title}.fchk'], capture_output=True, text=True)




