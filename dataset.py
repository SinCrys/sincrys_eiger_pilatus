from dataclasses import dataclass
import pandas as pd
from symmetry import Symmetry
import numpy as np

@dataclass
class dataset:
    symmetry: str = '1'
    name: str = 'new dataset'
    d = {}
    
    def __init__(self, path, skipfooter=17, symmetry='1'):
        """
        # load data
        """
        print(f'New dataset: {path}')
        self.data = self.read_hkl(path, skipfooter)
        self.name = path
        self.set_symmetry(symmetry)
        print(f'Length of data loaded: {len(self.data)}')
        
        # reduce the HKL's to families
        self.data['hkl'] = self.data['hkl'].apply(self.get_family)
        
        # average equivalent HKL's according to family
        self.data = self.data.groupby(by=self.data['hkl']).mean()
        
        # re-index
        self.data.reset_index()
        
        print(f'Length of reduced data: {len(self.data)}')
        
    def read_hkl(self, path: str, skipfooter: int = 17) -> pd.core.frame.DataFrame:
        """
         read a path .hkl file and return as a Pandas dataframe
        """
        df = pd.read_csv(path,
                         engine='python',
                         header=None,
                         delim_whitespace=True,
                         names=['h', 'k', 'l', 'intensity', 'sigma', 'unknown'], # I don't know exactly what the last field is
                         dtype={'h': int, 'k': int, 'l': int, 'intensity': float, 'd': float, 'doom': int},
                         skipfooter=skipfooter
                        )
        
        # merge the H,K,L columns into a single tuple-column
        df['hkl'] = df[['h','k','l']].apply(tuple, axis=1)
        
        # remove the others again
        for x in ['h', 'k', 'l', 'unknown']:
            df.pop(x)

        # TODO: reorder hkl to 0th column again?
        df.set_index('hkl')

        # return!
        return df
    
    def set_symmetry(self, symmetry='1') -> None:
        """
         get symmetry operations
        """
        self.symmetry = symmetry
        print(f'Setting symmetry to {symmetry} for {self.name}')
        self.symops = Symmetry[self.symmetry]
        print(f'No. of symops: {len(self.symops)}')
    
        
    def perform_symmetry(self, hkl) -> list:
        """
         perform symmetry operation on an hkl
         
         returns unique symmetry-equivalent hkl's
        """
        #print(hkl)
        # get the product of the input and the symmetry operations
        dotp = np.array(hkl).dot(self.symops)
        
        # find the unique hkl's from that
        unique = np.unique(dotp, axis=0)
        #print(f'Length of the dotproduct {len(dotp)} and length of unique ops: {len(unique)}')
        return unique
    
    def get_family(self, hkl) -> tuple:
        """
         get the unique HKL for the family
         when applying the symmetry op's on
         the hkl
        """
        # dotproduct gives the family
        dot = np.array(hkl).dot(self.symops)
        
        # get the unique as the last in the array
        unique = np.unique(dot, axis=0)[-1]
        
        # return as tuple HKL
        return tuple(unique)
    
    
        