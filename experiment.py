import sys
import logging
import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProductConfig:
    """Configuration for a single product"""
    product_id: str
    initial_inventory: float
    volume: float
    product_group: str
    brand: str
    
@dataclass
class SourcingConfig:
    """Configuration for sourcing channels"""
    jit_lead_time: int = 1  # JIT lead time in weeks
    llt_lead_time: int = 12  # Long lead time in weeks
    discount_factor: float = 0.95  # γ from paper

def dirichlet_sample(alpha_list):
    """Simple approximation of Dirichlet distribution sampling"""
    # Generate gamma samples
    gamma_samples = []
    for alpha in alpha_list:
        # Simplified gamma sampling using exponential distribution
        gamma_sample = random.expovariate(1.0) ** (1.0/alpha) if alpha > 0 else 0.001
        gamma_samples.append(gamma_sample)
    
    # Normalize
    total = sum(gamma_samples)
    if total > 0:
        return [g / total for g in gamma_samples]
    else:
        n = len(alpha_list)
        return [1.0/n for _ in alpha_list]

def normal_random(mu=0, sigma=1):
    """Generate normal random variable using Box-Muller transform"""
    if not hasattr(normal_random, "stored"):
        normal_random.stored = None
    
    if normal_random.stored is not None:
        result = normal_random.stored
        normal_random.stored = None
        return mu + sigma * result
    
    # Generate two uniform random numbers
    u1 = random.random()
    u2 = random.random()
    
    # Box-Muller transform
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    
    normal_random.stored = z1
    return mu + sigma * z0

class ExogenousProcess:
    """Models exogenous processes independent of buy policy"""
    
    def __init__(self, product_id: str, config: SourcingConfig = None):
        self.product_id = product_id
        self.config = config or SourcingConfig()
        self.demand_history = []
        self.price_history = []
        self.cost_history = []
        
    def generate_weekly_state(self, week: int) -> Dict:
        """Generate exogenous state for a given week"""
        try:
            # Simplified stochastic processes based on paper
            base_demand = 100 + 20 * math.sin(week * 2 * math.pi / 52)  # Seasonal pattern
            demand = max(0, normal_random(base_demand, base_demand * 0.3))
            
            price = 50 + 5 * normal_random()
            jit_cost = 30 + 3 * normal_random()
            llt_cost = 25 + 2 * normal_random()  # LLT has discount
            
            # Arrival shares (ρ) - simplified version
            jit_arrival_shares = self._generate_arrival_shares(self.config.jit_lead_time)
            llt_arrival_shares = self._generate_arrival_shares(self.config.llt_lead_time)
            
            # Vendor constraints (M) - simplified
            jit_constraints = {
                'min_order': 10,
                'batch_size': 5,
                'max_supply': max(0, 200 + 50 * normal_random())
            }
            llt_constraints = {
                'min_order': 50,
                'batch_size': 10,
                'max_supply': max(0, 500 + 100 * normal_random())
            }
            
            return {
                'demand': demand,
                'price': price,
                'jit_cost': jit_cost,
                'llt_cost': llt_cost,
                'jit_arrival_shares': jit_arrival_shares,
                'llt_arrival_shares': llt_arrival_shares,
                'jit_constraints': jit_constraints,
                'llt_constraints': llt_constraints
            }
        except Exception as e:
            logger.error(f"Error generating weekly state for week {week}: {e}")
            raise
    
    def _generate_arrival_shares(self, lead_time: int) -> List[float]:
        """Generate arrival share distribution"""
        alpha_list = [1.0] * (lead_time + 1)
        shares = dirichlet_sample(alpha_list)
        return shares

class OrderPostProcessor:
    """Processes orders according to vendor constraints (f_p in paper)"""
    
    @staticmethod
    def process_order(order_qty: float, constraints: Dict) -> float:
        """Apply vendor constraints to order quantity"""
        try:
            # Apply minimum order constraint
            if order_qty > 0 and order_qty < constraints['min_order']:
                order_qty = constraints['min_order']
            
            # Apply batch size constraint
            if constraints['batch_size'] > 0:
                batches = math.ceil(order_qty / constraints['batch_size'])
                order_qty = batches * constraints['batch_size']
                
            return float(order_qty)
        except Exception as e:
            logger.error(f"Error processing order: {e}")
            return 0.0

class ArrivalSimulator:
    """Simulates order arrivals (o in paper)"""
    
    @staticmethod
    def simulate_arrivals(
        processed_qty: float,
        max_supply: float,
        arrival_shares: List[float],
        week: int
    ) -> Dict[int, float]:
        """Simulate arrivals over lead time"""
        try:
            # Apply supply constraint (U in paper)
            actual_qty = min(processed_qty, max_supply)
            
            # Distribute over lead time
            arrivals = {}
            for lead_offset, share in enumerate(arrival_shares):
                arrivals[week + lead_offset] = actual_qty * share
                
            return arrivals
        except Exception as e:
            logger.error(f"Error simulating arrivals: {e}")
            return {}

class InventorySystem:
    """Manages inventory dynamics (equations 6-7 in paper)"""
    
    def __init__(self, product_config: ProductConfig):
        self.product_config = product_config
        self.current_inventory = product_config.initial_inventory
        self.inventory_history = [product_config.initial_inventory]
        self.pending_arrivals = {}  # week -> quantity
        
    def update_inventory(
        self,
        demand: float,
        jit_arrivals: Dict[int, float],
        llt_arrivals: Dict[int, float],
        week: int
    ) -> float:
        """Update inventory according to equations 6-7"""
        try:
            # Add arrivals to pending
            for arrival_week, qty in jit_arrivals.items():
                self.pending_arrivals[arrival_week] = \
                    self.pending_arrivals.get(arrival_week, 0) + qty
                    
            for arrival_week, qty in llt_arrivals.items():
                self.pending_arrivals[arrival_week] = \
                    self.pending_arrivals.get(arrival_week, 0) + qty
            
            # Add arrivals for current week
            current_arrivals = self.pending_arrivals.pop(week, 0)
            self.current_inventory += current_arrivals
            
            # Calculate inventory after demand (equation 7)
            inventory_after_demand = max(0, self.current_inventory - demand)
            self.current_inventory = inventory_after_demand
            self.inventory_history.append(self.current_inventory)
            
            return self.current_inventory
        except Exception as e:
            logger.error(f"Error updating inventory: {e}")
            return self.current_inventory

def matrix_multiply(A, B):
    """Simple matrix multiplication"""
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions don't match for multiplication")
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum_val = 0
            for k in range(len(B)):
                sum_val += A[i][k] * B[k][j]
            row.append(sum_val)
        result.append(row)
    return result

def vector_add(a, b):
    """Add two vectors"""
    return [a[i] + b[i] for i in range(len(a))]

def tanh(x):
    """Hyperbolic tangent function"""
    return math.tanh(x)

def vector_tanh(v):
    """Apply tanh to each element of vector"""
    return [tanh(x) for x in v]

class PolicyNetwork:
    """Deep RL policy network (π in paper)"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Initialize simple neural network weights
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # Initialize weights with small random values
            weight_matrix = []
            for i in range(prev_dim):
                row = [normal_random(0, 0.1) for _ in range(hidden_dim)]
                weight_matrix.append(row)
            self.weights.append(weight_matrix)
            
            # Initialize biases to zero
            self.biases.append([0.0 for _ in range(hidden_dim)])
            prev_dim = hidden_dim
            
        # Output layer (2 outputs: JIT and LLT orders)
        weight_matrix = []
        for i in range(prev_dim):
            row = [normal_random(0, 0.1) for _ in range(2)]
            weight_matrix.append(row)
        self.weights.append(weight_matrix)
        self.biases.append([0.0, 0.0])
        
        logger.info(f"Initialized policy network with architecture: "
                   f"{input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 2")
    
    def forward(self, state_features: List[float]) -> List[float]:
        """Forward pass through network"""
        try:
            x = [[f] for f in state_features]  # Convert to column vector
            
            for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
                # Matrix multiplication
                result = matrix_multiply(W, x)
                # Add bias and apply activation
                x = [[tanh(result[j][0] + b[j])] for j in range(len(result))]
                
            # Output layer (no activation for continuous actions)
            W_out = self.weights[-1]
            b_out = self.biases[-1]
            output = matrix_multiply(W_out, x)
            
            # Convert back to list and add bias
            output_list = [max(0, output[i][0] + b_out[i]) for i in range(len(output))]
            
            return output_list
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            return [0.0, 0.0]
    
    def get_parameters(self) -> List:
        """Get all network parameters"""
        return self.weights + self.biases
    
    def update_parameters(self, gradients: List, lr: float = 0.001):
        """Update parameters with gradients"""
        try:
            # Simplified parameter update
            for i in range(len(self.weights)):
                if i < len(gradients):
                    grad = gradients[i]
                    if isinstance(grad, (int, float)):
                        # Scalar gradient - apply to all weights
                        for j in range(len(self.weights[i])):
                            for k in range(len(self.weights[i][j])):
                                self.weights[i][j][k] -= lr * grad
                    elif isinstance(grad, list):
                        # Try to apply gradient if dimensions match
                        if len(grad) == len(self.weights[i]):
                            for j in range(len(self.weights[i])):
                                if isinstance(grad[j], list) and len(grad[j]) == len(self.weights[i][j]):
                                    for k in range(len(self.weights[i][j])):
                                        self.weights[i][j][k] -= lr * grad[j][k]
                                elif isinstance(grad[j], (int, float)):
                                    for k in range(len(self.weights[i][j])):
                                        self.weights[i][j][k] -= lr * grad[j]
                        
            for i in range(len(self.biases)):
                bias_idx = len(self.weights) + i
                if bias_idx < len(gradients):
                    grad = gradients[bias_idx]
                    if isinstance(grad, (int, float)):
                        # Scalar gradient - apply to all biases
                        for j in range(len(self.biases[i])):
                            self.biases[i][j] -= lr * grad
                    elif isinstance(grad, list) and len(grad) == len(self.biases[i]):
                        for j in range(len(self.biases[i])):
                            self.biases[i][j] -= lr * grad[j]
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")

class DualSourcingRL:
    """Main Dual Sourcing RL agent (Algorithm 1 in paper)"""
    
    def __init__(self, config: SourcingConfig):
        self.config = config
        self.products = {}
        self.policy_networks = {}
        
        # Training state
        self.total_rewards = []
        self.training_losses = []
        
    def add_product(self, product_config: ProductConfig):
        """Add a product to manage"""
        try:
            self.products[product_config.product_id] = {
                'config': product_config,
                'exogenous': ExogenousProcess(product_config.product_id, self.config),
                'inventory': InventorySystem(product_config),
                'order_processor': OrderPostProcessor(),
                'arrival_simulator': ArrivalSimulator()
            }
            
            # Initialize policy network for this product
            input_dim = 20  # Simplified feature dimension
            self.policy_networks[product_config.product_id] = PolicyNetwork(input_dim)
            
            logger.info(f"Added product {product_config.product_id} with initial inventory "
                       f"{product_config.initial_inventory}")
        except Exception as e:
            logger.error(f"Error adding product {product_config.product_id}: {e}")
            sys.exit(1)
    
    def extract_features(self, product_id: str, history: List[Dict], week: int) -> List[float]:
        """Extract features for policy network (simplified version)"""
        try:
            product_data = self.products[product_id]
            
            # Basic features based on paper Appendix C.1
            features = []
            
            # 1. Current inventory level
            features.append(product_data['inventory'].current_inventory)
            
            # 2. Time series features (simplified)
            if history:
                last_state = history[-1]
                features.append(last_state.get('demand', 0))
                features.append(last_state.get('price', 0))
                features.append(last_state.get('jit_cost', 0))
                features.append(last_state.get('llt_cost', 0))
            else:
                features.extend([0, 0, 0, 0])
            
            # 3. Seasonal features
            features.append(math.sin(week * 2 * math.pi / 52))
            features.append(math.cos(week * 2 * math.pi / 52))
            
            # 4. Product features (encoded)
            config = product_data['config']
            features.append(config.volume)
            # Simplified encoding of categorical features
            features.append(hash(config.product_group) % 100 / 100)
            features.append(hash(config.brand) % 100 / 100)
            
            # 5. Pending arrivals
            pending_total = sum(product_data['inventory'].pending_arrivals.values())
            features.append(pending_total)
            
            # Pad to expected dimension
            while len(features) < 20:
                features.append(0.0)
                
            return features[:20]
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return [0.0] * 20
    
    def compute_reward(
        self,
        product_id: str,
        demand: float,
        price: float,
        jit_cost: float,
        llt_cost: float,
        jit_arrivals: Dict[int, float],
        llt_arrivals: Dict[int, float],
        holding_cost: float = 0.1
    ) -> float:
        """Compute reward according to equation 8"""
        try:
            product_data = self.products[product_id]
            inventory = product_data['inventory'].current_inventory
            
            # Sales revenue
            sales = min(demand, inventory)
            revenue = price * sales
            
            # JIT ordering costs
            jit_total_arrivals = sum(jit_arrivals.values())
            jit_ordering_cost = jit_cost * jit_total_arrivals
            
            # LLT ordering costs
            llt_total_arrivals = sum(llt_arrivals.values())
            llt_ordering_cost = llt_cost * llt_total_arrivals
            
            # Holding cost
            holding_cost_total = holding_cost * inventory
            
            # Total reward
            reward = revenue - jit_ordering_cost - llt_ordering_cost - holding_cost_total
            
            return reward
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return 0.0
    
    def train_episode(self, num_weeks: int = 52, batch_size: int = 10) -> float:
        """Train for one episode (Algorithm 1)"""
        logger.info("Starting training episode...")
        
        try:
            episode_reward = 0
            product_ids = list(self.products.keys())
            
            if len(product_ids) == 0:
                logger.error("No products configured for training")
                return 0
            
            # Sample batch of products
            batch_products = random.sample(product_ids, min(batch_size, len(product_ids)))
            
            batch_rewards = []
            gradients_accumulated = {pid: [] for pid in batch_products}
            
            for product_id in batch_products:
                product_data = self.products[product_id]
                policy_net = self.policy_networks[product_id]
                
                product_reward = 0
                history = []
                
                for week in range(num_weeks):
                    try:
                        # Get exogenous state
                        exogenous_state = product_data['exogenous'].generate_weekly_state(week)
                        
                        # Extract features
                        features = self.extract_features(product_id, history, week)
                        
                        # Get action from policy
                        action = policy_net.forward(features)
                        jit_order = action[0]
                        llt_order = action[1]
                        
                        # Process orders through vendor constraints
                        processed_jit = product_data['order_processor'].process_order(
                            jit_order, exogenous_state['jit_constraints']
                        )
                        processed_llt = product_data['order_processor'].process_order(
                            llt_order, exogenous_state['llt_constraints']
                        )
                        
                        # Simulate arrivals
                        jit_arrivals = product_data['arrival_simulator'].simulate_arrivals(
                            processed_jit,
                            exogenous_state['jit_constraints']['max_supply'],
                            exogenous_state['jit_arrival_shares'],
                            week
                        )
                        
                        llt_arrivals = product_data['arrival_simulator'].simulate_arrivals(
                            processed_llt,
                            exogenous_state['llt_constraints']['max_supply'],
                            exogenous_state['llt_arrival_shares'],
                            week
                        )
                        
                        # Update inventory
                        product_data['inventory'].update_inventory(
                            exogenous_state['demand'],
                            jit_arrivals,
                            llt_arrivals,
                            week
                        )
                        
                        # Compute reward
                        reward = self.compute_reward(
                            product_id,
                            exogenous_state['demand'],
                            exogenous_state['price'],
                            exogenous_state['jit_cost'],
                            exogenous_state['llt_cost'],
                            jit_arrivals,
                            llt_arrivals
                        )
                        
                        # Discount reward
                        discounted_reward = reward * (self.config.discount_factor ** week)
                        product_reward += discounted_reward
                        
                        # Store history
                        history.append({
                            'week': week,
                            'state': exogenous_state,
                            'action': action,
                            'reward': reward,
                            'inventory': product_data['inventory'].current_inventory
                        })
                        
                        # Simplified gradient computation (REINFORCE style)
                        log_prob = math.log(abs(action[0]) + 1e-8) + math.log(abs(action[1]) + 1e-8)
                        gradient = discounted_reward * log_prob
                        
                        if len(gradients_accumulated[product_id]) == 0:
                            gradients_accumulated[product_id] = [gradient]
                        else:
                            gradients_accumulated[product_id][0] += gradient
                            
                    except Exception as e:
                        logger.error(f"Error during training week {week} for product {product_id}: {e}")
                        continue
                
                batch_rewards.append(product_reward)
                episode_reward += product_reward
                
                logger.info(f"Product {product_id}: Total reward = {product_reward:.2f}, "
                           f"Final inventory = {product_data['inventory'].current_inventory:.2f}")
            
            # Update policies (simplified)
            for product_id in batch_products:
                if gradients_accumulated[product_id]:
                    avg_gradient = gradients_accumulated[product_id][0] / num_weeks
                    # Simplified update - in practice would compute proper gradients
                    self.policy_networks[product_id].update_parameters(
                        [avg_gradient, avg_gradient],
                        lr=0.001
                    )
            
            avg_batch_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
            self.total_rewards.append(episode_reward)
            
            logger.info(f"Training episode completed. Average batch reward: {avg_batch_reward:.2f}, "
                       f"Total episode reward: {episode_reward:.2f}")
            
            return episode_reward
            
        except Exception as e:
            logger.error(f"Critical error during training episode: {e}")
            sys.exit(1)
    
    def evaluate(self, num_weeks: int = 12) -> Dict:
        """Evaluate current policy"""
        logger.info("Starting evaluation...")
        
        try:
            results = {
                'total_reward': 0,
                'product_results': {},
                'avg_inventory': 0,
                'service_level': 0
            }
            
            total_demand = 0
            total_sales = 0
            
            for product_id, product_data in self.products.items():
                product_reward = 0
                product_history = []
                product_demand = 0
                product_sales = 0
                
                # Reset inventory for evaluation
                product_data['inventory'].current_inventory = product_data['config'].initial_inventory
                product_data['inventory'].pending_arrivals = {}
                
                for week in range(num_weeks):
                    try:
                        # Get exogenous state
                        exogenous_state = product_data['exogenous'].generate_weekly_state(week)
                        
                        # Extract features
                        features = self.extract_features(product_id, product_history, week)
                        
                        # Get action from policy (no exploration during evaluation)
                        action = self.policy_networks[product_id].forward(features)
                        jit_order = action[0]
                        llt_order = action[1]
                        
                        # Process orders
                        processed_jit = product_data['order_processor'].process_order(
                            jit_order, exogenous_state['jit_constraints']
                        )
                        processed_llt = product_data['order_processor'].process_order(
                            llt_order, exogenous_state['llt_constraints']
                        )
                        
                        # Simulate arrivals
                        jit_arrivals = product_data['arrival_simulator'].simulate_arrivals(
                            processed_jit,
                            exogenous_state['jit_constraints']['max_supply'],
                            exogenous_state['jit_arrival_shares'],
                            week
                        )
                        
                        llt_arrivals = product_data['arrival_simulator'].simulate_arrivals(
                            processed_llt,
                            exogenous_state['llt_constraints']['max_supply'],
                            exogenous_state['llt_arrival_shares'],
                            week
                        )
                        
                        # Update inventory
                        current_inv = product_data['inventory'].current_inventory
                        product_data['inventory'].update_inventory(
                            exogenous_state['demand'],
                            jit_arrivals,
                            llt_arrivals,
                            week
                        )
                        
                        # Calculate sales and demand for service level
                        demand = exogenous_state['demand']
                        sales = min(demand, current_inv)
                        product_demand += demand
                        product_sales += sales
                        total_demand += demand
                        total_sales += sales
                        
                        # Compute reward
                        reward = self.compute_reward(
                            product_id,
                            demand,
                            exogenous_state['price'],
                            exogenous_state['jit_cost'],
                            exogenous_state['llt_cost'],
                            jit_arrivals,
                            llt_arrivals
                        )
                        
                        product_reward += reward
                        
                        # Store history
                        product_history.append({
                            'week': week,
                            'state': exogenous_state,
                            'action': action,
                            'reward': reward,
                            'inventory': product_data['inventory'].current_inventory
                        })
                        
                    except Exception as e:
                        logger.error(f"Error during evaluation week {week} for product {product_id}: {e}")
                        continue
                
                # Calculate product-level metrics
                product_service_level = (product_sales / product_demand) if product_demand > 0 else 0
                avg_inventory = sum([h['inventory'] for h in product_history]) / len(product_history) if product_history else 0
                
                results['product_results'][product_id] = {
                    'reward': product_reward,
                    'service_level': product_service_level,
                    'avg_inventory': avg_inventory,
                    'total_demand': product_demand,
                    'total_sales': product_sales
                }
                
                results['total_reward'] += product_reward
                
                logger.info(f"Product {product_id} evaluation: Reward={product_reward:.2f}, "
                           f"Service Level={product_service_level:.3f}, Avg Inventory={avg_inventory:.2f}")
            
            # Calculate overall metrics
            results['service_level'] = (total_sales / total_demand) if total_demand > 0 else 0
            avg_inventories = [pr['avg_inventory'] for pr in results['product_results'].values()]
            results['avg_inventory'] = sum(avg_inventories) / len(avg_inventories) if avg_inventories else 0
            
            logger.info(f"Overall evaluation: Total Reward={results['total_reward']:.2f}, "
                       f"Service Level={results['service_level']:.3f}, Avg Inventory={results['avg_inventory']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error during evaluation: {e}")
            sys.exit(1)

def main():
    """Main function to run the experiment"""
    try:
        logger.info("Starting Dual Sourcing RL Experiment")
        
        # Configuration
        sourcing_config = SourcingConfig(
            jit_lead_time=1,
            llt_lead_time=12,
            discount_factor=0.95
        )
        
        # Initialize RL agent
        agent = DualSourcingRL(sourcing_config)
        
        # Add sample products
        products = [
            ProductConfig("PROD001", 50.0, 1.0, "GROUP_A", "BRAND_X"),
            ProductConfig("PROD002", 75.0, 1.5, "GROUP_B", "BRAND_Y"),
            ProductConfig("PROD003", 100.0, 2.0, "GROUP_A", "BRAND_Z")
        ]
        
        for product in products:
            agent.add_product(product)
        
        # Training phase
        logger.info("Starting training phase...")
        num_training_episodes = 5
        training_results = []
        
        for episode in range(num_training_episodes):
            logger.info(f"Training episode {episode + 1}/{num_training_episodes}")
            episode_reward = agent.train_episode(num_weeks=26, batch_size=2)
            training_results.append(episode_reward)
        
        # Evaluation phase
        logger.info("Starting evaluation phase...")
        evaluation_results = agent.evaluate(num_weeks=12)
        
        # Results summary
        logger.info("=== EXPERIMENT RESULTS ===")
        logger.info(f"Training episodes completed: {num_training_episodes}")
        logger.info(f"Average training reward: {sum(training_results)/len(training_results):.2f}")
        logger.info(f"Final evaluation reward: {evaluation_results['total_reward']:.2f}")
        logger.info(f"Overall service level: {evaluation_results['service_level']:.3f}")
        logger.info(f"Average inventory level: {evaluation_results['avg_inventory']:.2f}")
        
        # Product-specific results
        for product_id, results in evaluation_results['product_results'].items():
            logger.info(f"{product_id}: Reward={results['reward']:.2f}, "
                       f"Service Level={results['service_level']:.3f}")
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()