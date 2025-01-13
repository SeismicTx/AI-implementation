import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from models.esm_antibody_transformer import ESMAntibodyTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for sequence generation"""
    target_kd: float  # Target KD in nM
    target_tm1: float  # Target melting temperature 1
    target_poi: float  # Target % POI

    # Generation parameters
    population_size: int = 1000
    num_iterations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # Sequence constraints
    min_length_vh: int = 110
    max_length_vh: int = 130
    min_length_vl: int = 105
    max_length_vl: int = 125

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

        # Amino acid vocabulary (matching your dataset)
        self.aa_vocab = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.idx_to_aa = {idx: aa for aa, idx in self.aa_vocab.items()}

    def _sequence_to_tensor(self, sequence: str) -> torch.Tensor:
        """Convert amino acid sequence to tensor"""
        return torch.tensor([self.aa_vocab[aa] for aa in sequence],
                          dtype=torch.long,
                          device=self.device)

    def _tensor_to_sequence(self, tensor: torch.Tensor) -> str:
        """Convert tensor back to amino acid sequence"""
        return ''.join(self.idx_to_aa[idx.item()] for idx in tensor)

    @torch.no_grad()
    def predict_properties(self, vh_seq: str, vl_seq: str) -> Dict[str, float]:
        """Predict antibody properties using the model"""
        vh_tensor = self._sequence_to_tensor(vh_seq)
        vl_tensor = self._sequence_to_tensor(vl_seq)

        #output = self.model(vh_tensor, vl_tensor)
        combined_tensor = pad_sequence([vh_tensor, vl_tensor])
        output = self.model(combined_tensor.unsqueeze(0))

        # Convert predictions back to original scale
        scaled_output = self.exp_scaler.inverse_transform(output.cpu().numpy())

        return {
            'KD': scaled_output[0][0],
            'Tm1': scaled_output[0][1],
            'POI': scaled_output[0][2]
        }

    def _initialize_sequence(self, length: int) -> str:
        """Generate random amino acid sequence"""
        return ''.join(random.choice(list(self.aa_vocab.keys()))
                      for _ in range(length))

    def _initialize_population(self) -> List[Tuple[str, str]]:
        """Generate initial population of VH, VL pairs"""
        population = []

        for _ in range(self.config.population_size):
            vh_length = random.randint(self.config.min_length_vh,
                                     self.config.max_length_vh)
            vl_length = random.randint(self.config.min_length_vl,
                                     self.config.max_length_vl)

            vh = self._initialize_sequence(vh_length)
            vl = self._initialize_sequence(vl_length)

            population.append((vh, vl))

        return population

    def _mutate_sequence(self, sequence: str) -> str:
        """Apply random mutations to sequence"""
        sequence = list(sequence)
        for i in range(len(sequence)):
            if random.random() < self.config.mutation_rate:
                sequence[i] = random.choice(list(self.aa_vocab.keys()))
        return ''.join(sequence)

    def _crossover(self, pair1: Tuple[str, str], pair2: Tuple[str, str]) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Perform crossover between two antibody pairs"""
        vh1, vl1 = pair1
        vh2, vl2 = pair2

        if random.random() < self.config.crossover_rate:
            # Crossover VH
            point_vh = random.randint(0, min(len(vh1), len(vh2)))
            new_vh1 = vh1[:point_vh] + vh2[point_vh:]
            new_vh2 = vh2[:point_vh] + vh1[point_vh:]

            # Crossover VL
            point_vl = random.randint(0, min(len(vl1), len(vl2)))
            new_vl1 = vl1[:point_vl] + vl2[point_vl:]
            new_vl2 = vl2[:point_vl] + vl1[point_vl:]

            return (new_vh1, new_vl1), (new_vh2, new_vl2)

        return pair1, pair2

    def _fitness_function(self, vh: str, vl: str) -> float:
        """Calculate fitness score based on predicted properties"""

        # Get predictions
        props = self.predict_properties(vh, vl)

        # Calculate fitness based on distance to targets
        kd_error = abs(props['KD'] - self.config.target_kd) / self.config.target_kd
        tm1_error = abs(props['Tm1'] - self.config.target_tm1) / self.config.target_tm1
        poi_error = abs(props['POI'] - self.config.target_poi) / self.config.target_poi

        # Weighted sum of errors (adjust weights based on importance)
        total_error = (kd_error * 0.4 +
                      tm1_error * 0.3 +
                      poi_error * 0.3)

        return 1.0 / (1.0 + total_error)

    def generate_sequences(self) -> List[Dict]:
        """
        Generate optimized antibody sequences

        Returns:
            List of dictionaries containing VH, VL sequences and predicted properties
        """
        # Initialize population
        population = self._initialize_population()

        best_fitness = 0
        best_solution = None

        for iteration in tqdm(range(self.config.num_iterations)):
            # Evaluate fitness
            fitness_scores = [
                self._fitness_function(vh, vl)
                for vh, vl in population
            ]

            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_solution = population[max_fitness_idx]

            # Selection
            selection_probs = np.array(fitness_scores) / sum(fitness_scores)
            selection_probs = np.nan_to_num(selection_probs, nan=1/len(fitness_scores))
            parent_indices = np.random.choice(
                len(population),
                size=len(population),
                p=selection_probs
            )
            parents = [population[i] for i in parent_indices]

            # Create next generation
            next_generation = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i], parents[i+1])

                    # Mutate children
                    child1 = (self._mutate_sequence(child1[0]),
                            self._mutate_sequence(child1[1]))
                    child2 = (self._mutate_sequence(child2[0]),
                            self._mutate_sequence(child2[1]))

                    next_generation.extend([child1, child2])

            population = next_generation

            logger.info(f"Iteration {iteration}, Best Fitness: {best_fitness:.4f}")

        # Return top sequences
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

        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)

        return results[:10]  # Return top 10 sequences
