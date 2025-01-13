import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from dataclasses import dataclass
import logging
from typing import List, Dict, Tuple, Optional, Set
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from models.antibody_transformer import AntibodyTransformer
from anarci import run_anarci

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for sequence generation"""
    target_kd: float  # Target KD in nM
    target_tm1: float  # Target melting temperature 1
    target_poi: float  # Target % POI

    # Generation parameters
    population_size: int = 10
    num_iterations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # Sequence constraints
    min_length_vh: int = 110
    max_length_vh: int = 130
    min_length_vl: int = 105
    max_length_vl: int = 125

    # Template sequences 
    vh_template: str = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINTKGLTNYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKGWFDYWGQGTLVTVSS" 
    vl_template: str = "DIQMTQSPSSLSASVGDRVTITCRASQGISNYLNWYQQKPGKAPKLLIYYASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPYTFGQGTKVEIK"   

class AntibodyGenerator:
    def __init__(self,
                 model: nn.Module,
                 config: GenerationConfig,
                 exp_scaler):
        """
        Initialize generator with trained model

        Args:
            model: Trained AntibodyPropertyPredictor
            config: Generation configuration
            exp_scaler: StandardScaler for experimental features
        """
        self.model = model
        self.config = config
        self.exp_scaler = exp_scaler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Amino acid vocabulary
        self.aa_vocab = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.idx_to_aa = {idx: aa for aa, idx in self.aa_vocab.items()}

        # Initialize CDR position cache
        self.cdr_cache = {}

    def _get_cdr_positions(self, sequence: str, chain_type: str) -> Set[int]:
        """
        Use ANARCI to identify CDR positions in the sequence
        
        Args:
            sequence: Amino acid sequence
            chain_type: 'H' for heavy chain or 'L' for light chain
        
        Returns:
            Set of positions (0-based indices) that are in CDRs
        """
        # Check cache first
        cache_key = (sequence, chain_type)
        if cache_key in self.cdr_cache:
            return self.cdr_cache[cache_key]

        # Run ANARCI
        numbering = run_anarci([('seq', sequence)], scheme='kabat', output=False)
        
        if not numbering or not numbering[0][0]:
            logger.warning(f"ANARCI failed to number {chain_type} chain sequence")
            return set()

        # Extract CDR positions based on Kabat numbering
        cdr_ranges = {
            'H': [
                (31, 35),   # CDR-H1
                (50, 65),   # CDR-H2
                (95, 102)   # CDR-H3
            ],
            'L': [
                (24, 34),   # CDR-L1
                (50, 56),   # CDR-L2
                (89, 97)    # CDR-L3
            ]
        }

        numbering_map = numbering[1][0][0][0]  # Get numbering for first sequence
        cdr_positions = set()
        
        for start, end in cdr_ranges[chain_type]:
            for pos in range(len(sequence)):
                if pos < len(numbering_map):
                    num = numbering_map[pos][0][0]
                    if start <= num <= end:
                        cdr_positions.add(pos)

        # Cache result
        self.cdr_cache[cache_key] = cdr_positions
        return cdr_positions

    def _mutate_sequence(self, sequence: str, chain_type: str) -> str:
        """Apply random mutations only to CDR regions"""
        cdr_positions = self._get_cdr_positions(sequence, chain_type)
        
        if not cdr_positions:
            logger.warning(f"No CDR positions identified for {chain_type} chain, skipping mutation")
            return sequence

        sequence = list(sequence)
        for pos in cdr_positions:
            if random.random() < self.config.mutation_rate:
                sequence[pos] = random.choice(list(self.aa_vocab.keys()))
        return ''.join(sequence)

    def _crossover(self, pair1: Tuple[str, str], pair2: Tuple[str, str]) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Perform crossover between two antibody pairs, preserving framework regions"""
        vh1, vl1 = pair1
        vh2, vl2 = pair2

        if random.random() < self.config.crossover_rate:
            # Get CDR positions
            vh1_cdrs = self._get_cdr_positions(vh1, 'H')
            vh2_cdrs = self._get_cdr_positions(vh2, 'H')
            vl1_cdrs = self._get_cdr_positions(vl1, 'L')
            vl2_cdrs = self._get_cdr_positions(vl2, 'L')

            # Create new sequences starting with templates
            new_vh1 = list(vh1)
            new_vh2 = list(vh2)
            new_vl1 = list(vl1)
            new_vl2 = list(vl2)

            # Crossover VH CDRs
            for pos in vh1_cdrs.union(vh2_cdrs):
                if random.random() < 0.5:
                    if pos < len(vh1) and pos < len(vh2):
                        new_vh1[pos], new_vh2[pos] = vh2[pos], vh1[pos]

            # Crossover VL CDRs
            for pos in vl1_cdrs.union(vl2_cdrs):
                if random.random() < 0.5:
                    if pos < len(vl1) and pos < len(vl2):
                        new_vl1[pos], new_vl2[pos] = vl2[pos], vl1[pos]

            return (''.join(new_vh1), ''.join(new_vl1)), (''.join(new_vh2), ''.join(new_vl2))

        return pair1, pair2

    def _initialize_sequence(self, template: str, length: int, chain_type: str) -> str:
        """Generate sequence based on template, randomizing CDRs"""
        # Start with template
        sequence = list(template[:length])
        
        # Randomize CDR positions
        cdr_positions = self._get_cdr_positions(template, chain_type)
        for pos in cdr_positions:
            if pos < length:
                sequence[pos] = random.choice(list(self.aa_vocab.keys()))
        
        return ''.join(sequence)

    def _initialize_population(self) -> List[Tuple[str, str]]:
        """Generate initial population of VH, VL pairs based on templates"""
        population = []

        for _ in range(self.config.population_size):
            vh_length = random.randint(self.config.min_length_vh,
                                     self.config.max_length_vh)
            vl_length = random.randint(self.config.min_length_vl,
                                     self.config.max_length_vl)

            vh = self._initialize_sequence(self.config.vh_template, vh_length, 'H')
            vl = self._initialize_sequence(self.config.vl_template, vl_length, 'L')

            population.append((vh, vl))

        return population

    # Rest of the methods remain the same as in original implementation
    def _sequence_to_tensor(self, sequence: str) -> torch.Tensor:
        return torch.tensor([self.aa_vocab[aa] for aa in sequence],
                          dtype=torch.long,
                          device=self.device)

    def _tensor_to_sequence(self, tensor: torch.Tensor) -> str:
        return ''.join(self.idx_to_aa[idx.item()] for idx in tensor)

    @torch.no_grad()
    def predict_properties(self, vh_seq: str, vl_seq: str) -> Dict[str, float]:
        vh_tensor = self._sequence_to_tensor(vh_seq)
        vl_tensor = self._sequence_to_tensor(vl_seq)
        concat = torch.transpose(pad_sequence([vh_tensor, vl_tensor]),0,1).unsqueeze(0)
        
        output = self.model(concat)
        scaled_output = self.exp_scaler.inverse_transform(output.cpu().numpy())
        return {
            'KD': scaled_output[0][0],
            'Tm1': scaled_output[0][1],
            'POI': scaled_output[0][2]
        }

    def _fitness_function(self, vh: str, vl: str) -> float:
        props = self.predict_properties(vh, vl)
        kd_error = abs(props['KD'] - self.config.target_kd) / self.config.target_kd
        tm1_error = abs(props['Tm1'] - self.config.target_tm1) / self.config.target_tm1
        poi_error = abs(props['POI'] - self.config.target_poi) / self.config.target_poi
        total_error = (kd_error * 0.4 + tm1_error * 0.3 + poi_error * 0.3)
        return 1.0 / (1.0 + total_error)

    def generate_sequences(self) -> List[Dict]:
        population = self._initialize_population()
        best_fitness = 0
        best_solution = None

        for iteration in tqdm(range(self.config.num_iterations)):
            fitness_scores = [
                self._fitness_function(vh, vl)
                for vh, vl in population
            ]

            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_solution = population[max_fitness_idx]

            selection_probs = np.array(fitness_scores) / sum(fitness_scores)
            selection_probs = np.nan_to_num(selection_probs, nan = 1/len(fitness_scores))

            parent_indices = np.random.choice(
                len(population),
                size=len(population),
                p=selection_probs
            )
            parents = [population[i] for i in parent_indices]

            next_generation = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i], parents[i+1])
                    child1 = (self._mutate_sequence(child1[0], 'H'),
                            self._mutate_sequence(child1[1], 'L'))
                    child2 = (self._mutate_sequence(child2[0], 'H'),
                            self._mutate_sequence(child2[1], 'L'))
                    next_generation.extend([child1, child2])

            population = next_generation
            logger.info(f"Iteration {iteration}, Best Fitness: {best_fitness:.4f}")

        results = []
        for vh, vl in population:
            properties = self.predict_properties(vh, vl)
            fitness = self._fitness_function(vh, vl)
            results.append({
                'VH': vh,
                'VL': vl,
                'properties': properties,
                'fitness': fitness
            })

        results.sort(key=lambda x: x['fitness'], reverse=True)
        return results[:10]